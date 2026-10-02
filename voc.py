"""PASCAL VOC 2012 dataset for object detection."""

from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

VOC_CLASSES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)

CLASS_TO_IDX = {name: i + 1 for i, name in enumerate(VOC_CLASSES)}  # 0 = background
IDX_TO_CLASS = {i: name for name, i in CLASS_TO_IDX.items()}
NUM_CLASSES = len(VOC_CLASSES) + 1


def resolve_voc_root(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir).expanduser().resolve()
    candidates = [
        data_dir,
        data_dir / "VOC2012",
        data_dir / "VOCdevkit" / "VOC2012",
    ]
    for candidate in candidates:
        if (candidate / "JPEGImages").is_dir() and (candidate / "Annotations").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find PASCAL VOC 2012.\n"
        f"Looked under: {data_dir}\n"
        "Expected a folder that contains JPEGImages/ and Annotations/ "
        "(or VOCdevkit/VOC2012 with those folders)."
    )


def _read_split_ids(voc_root: Path, split: str) -> list[str]:
    split_file = voc_root / "ImageSets" / "Main" / f"{split}.txt"
    if split_file.is_file():
        return [line.strip() for line in split_file.read_text().splitlines() if line.strip()]
    ids = sorted(p.stem for p in (voc_root / "JPEGImages").glob("*.jpg"))
    cut = max(1, int(0.9 * len(ids)))
    return ids[:cut] if split == "train" else ids[cut:]


def _parse_xml(xml_path: Path, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    root = ET.parse(xml_path).getroot()
    width, height = image_size
    boxes = []
    labels = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_IDX:
            continue
        if obj.findtext("difficult") == "1":
            continue
        box = obj.find("bndbox")
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        xmin = min(max(xmin, 0.0), width - 1)
        ymin = min(max(ymin, 0.0), height - 1)
        xmax = min(max(xmax, 1.0), width)
        ymax = min(max(ymax, 1.0), height)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_IDX[name])

    if boxes:
        return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)
    return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)


class VOCDetection(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train"):
        self.voc_root = resolve_voc_root(data_dir)
        self.ids = _read_split_ids(self.voc_root, split)
        self.augment = split == "train"
        self.image_dir = self.voc_root / "JPEGImages"
        self.anno_dir = self.voc_root / "Annotations"

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, index: int):
        image_id = self.ids[index]
        image = Image.open(self.image_dir / f"{image_id}.jpg").convert("RGB")
        boxes, labels = _parse_xml(self.anno_dir / f"{image_id}.xml", image.size)

        image = TF.to_tensor(image)
        if self.augment and torch.rand(1).item() < 0.5 and boxes.numel() > 0:
            image = TF.hflip(image)
            width = image.shape[-1]
            boxes[:, [0, 2]] = width - boxes[:, [2, 0]]

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([index]),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
