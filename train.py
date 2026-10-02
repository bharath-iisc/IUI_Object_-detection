"""Train Faster R-CNN on PASCAL VOC 2012.

Example:
    python train.py --data-dir /path/to/VOCdevkit --epochs 10
"""

from pathlib import Path
import argparse
import random

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import get_model
from voc import VOCDetection, collate_fn


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, loader, optimizer, device) -> float:
    model.train()
    running_loss = 0.0
    for step, (images, targets) in enumerate(tqdm(loader, desc="train", leave=False), start=1):
        images = [image.to(device) for image in images]
        targets = [{key: value.to(device) for key, value in target.items()} for target in targets]

        loss_dict = model(images, targets)
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if step % 50 == 0 or step == len(loader):
            print(f"  step {step}/{len(loader)}  loss={loss.item():.4f}", flush=True)

    return running_loss / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on VOC 2012")
    parser.add_argument("--data-dir", type=str, default="data", help="Folder containing VOC2012 or VOCdevkit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"device: {device}")

    train_dataset = VOCDetection(args.data_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
    )
    print(f"train images: {len(train_dataset)}")

    model = get_model(freeze_backbone=args.freeze_backbone).to(device)
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_path = out_dir / "fasterrcnn_voc2012.pth"

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "num_classes": model.roi_heads.box_predictor.cls_score.out_features,
            },
            weights_path,
        )
        print(f"epoch {epoch}/{args.epochs}  loss={loss:.4f}  saved={weights_path}")


if __name__ == "__main__":
    main()
