"""Run the trained VOC detector on an image or a folder of images.

Example:
    python predict.py --weights checkpoints/fasterrcnn_voc2012.pth --image photo.jpg
"""

from pathlib import Path
import argparse

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms import functional as TF

from model import get_model
from voc import IDX_TO_CLASS

COLORS = [
    (220, 20, 60),
    (0, 128, 255),
    (50, 205, 50),
    (255, 140, 0),
    (148, 0, 211),
    (255, 215, 0),
    (0, 206, 209),
    (255, 20, 147),
]


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(weights: str, device: torch.device):
    checkpoint = torch.load(weights, map_location=device)
    model = get_model()
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def draw_detections(image: Image.Image, prediction: dict, score_threshold: float) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("Arial.ttf", 14)
    except OSError:
        font = ImageFont.load_default()

    keep = prediction["scores"] >= score_threshold
    boxes = prediction["boxes"][keep].tolist()
    labels = prediction["labels"][keep].tolist()
    scores = prediction["scores"][keep].tolist()

    for box, label, score in zip(boxes, labels, scores):
        color = COLORS[label % len(COLORS)]
        name = IDX_TO_CLASS.get(label, str(label))
        xmin, ymin, xmax, ymax = box
        draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)
        caption = f"{name} {score:.2f}"
        draw.text((xmin + 2, max(ymin - 16, 0)), caption, fill=color, font=font)
    return image


@torch.no_grad()
def predict_image(model, image_path: Path, device: torch.device, score_threshold: float) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    tensor = TF.to_tensor(image).to(device)
    prediction = model([tensor])[0]
    return draw_detections(image, prediction, score_threshold)


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in path.iterdir() if p.suffix.lower() in extensions)


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect VOC objects in images")
    parser.add_argument("--weights", type=str, default="checkpoints/fasterrcnn_voc2012.pth")
    parser.add_argument("--image", type=str, required=True, help="Image file or folder")
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="outputs")
    args = parser.parse_args()

    device = get_device()
    model = load_model(args.weights, device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(Path(args.image))
    if not images:
        raise FileNotFoundError(f"No images found at {args.image}")

    for image_path in images:
        result = predict_image(model, image_path, device, args.score)
        save_path = out_dir / f"{image_path.stem}_pred{image_path.suffix}"
        result.save(save_path)
        print(f"saved {save_path}")


if __name__ == "__main__":
    main()
