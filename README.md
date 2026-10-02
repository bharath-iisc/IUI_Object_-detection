# Object Detection with a Pre-trained Backbone

Faster R-CNN on **PASCAL VOC 2012**, using an **ImageNet-pretrained ResNet-50** as the backbone.

The detector is not trained from scratch. ResNet-50 first learns generic vision on ImageNet (~1.2M images). A Faster R-CNN head is then fine-tuned on VOC so it can draw boxes around 20 object classes.

## Pipeline

```
JPEG + XML  →  voc.py  →  image tensor + boxes
                              ↓
                         model.py
              ImageNet ResNet-50 + FPN + RPN + detection head
                              ↓
                         train.py   (10 epochs on VOC train)
                              ↓
                        predict.py  (draw boxes on new photos)
```

## Dataset

PASCAL VOC 2012, 20 classes:

```
aeroplane, bicycle, bird, boat, bottle,
bus, car, cat, chair, cow,
diningtable, dog, horse, motorbike, person,
pottedplant, sheep, sofa, train, tvmonitor
```

| Split | Images |
|---|---|
| Train (`ImageSets/Main/train.txt`) | 5,717 |
| Val (`val.txt`) | 5,823 |

This copy lives at:

```
Data/archive/VOC2012_train_val/VOC2012_train_val/
  JPEGImages/
  Annotations/
  ImageSets/Main/
```

`--data-dir` can be that folder, or any parent that contains `JPEGImages/` and `Annotations/`. Training uses `train.txt` only. Extra JPEGs on disk that are not in that split are ignored.

## Setup

Python 3.10+ (this run used 3.14). Apple GPU (MPS), NVIDIA CUDA, or CPU.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt`: `torch`, `torchvision`, `Pillow`, `tqdm`.

The first training run downloads ImageNet ResNet-50 weights into the PyTorch hub cache (`~/.cache/torch/hub/checkpoints/`).

## Train

```bash
python train.py \
  --data-dir Data/archive/VOC2012_train_val/VOC2012_train_val \
  --epochs 10 \
  --batch-size 2
```

Useful flags:

| Flag | Default | Meaning |
|---|---|---|
| `--epochs` | 10 | Passes over the train split |
| `--batch-size` | 2 | Images per step (keep small on a laptop) |
| `--lr` | 0.005 | SGD learning rate |
| `--freeze-backbone` | off | If set, ImageNet weights stay fixed |
| `--out-dir` | `checkpoints` | Where to write the `.pth` file |

Checkpoints are overwritten after every epoch:

```
checkpoints/fasterrcnn_voc2012.pth
```

Logs go to `logs/train.log` if you tee the command.

### Result of the 10-epoch run

Device: Apple MPS. About 5.4 hours. Loss:

| Epoch | Loss |
|---|---|
| 1 | 0.48 |
| 2 | 0.39 |
| 3 | 0.36 |
| 4 | 0.29 |
| 5 | 0.28 |
| 6 | 0.27 |
| 7 | 0.26 |
| 8 | 0.26 |
| 9 | 0.26 |
| 10 | **0.26** |

## Predict

```bash
python predict.py \
  --weights checkpoints/fasterrcnn_voc2012.pth \
  --image path/to/photo.jpg
```

`--image` can be a single file or a folder of images. Boxes with score ≥ 0.5 are drawn (change with `--score`). Output:

```
outputs/<name>_pred.jpg
```

Example on a VOC test photo:

```bash
python predict.py \
  --weights checkpoints/fasterrcnn_voc2012.pth \
  --image Data/archive/VOC2012_test/VOC2012_test/JPEGImages/2008_000001.jpg
```

## Project layout

```
voc.py            VOC 2012 loader (images, XML boxes, train/val split)
model.py          Faster R-CNN + ImageNet ResNet-50 backbone
train.py          Fine-tune on the VOC train split
predict.py        Draw boxes on new images
requirements.txt
README.md         This file
Data/             VOC 2012 download
checkpoints/      Trained weights
outputs/          Prediction images
logs/             Training log
```

## How the pieces fit

- **`voc.py`** reads `JPEGImages/` and `Annotations/`, maps the 20 names to ids `1–20` (id `0` is background), and applies a random horizontal flip when `split="train"`.
- **`model.py`** builds Faster R-CNN with `weights=None` (no COCO detector) and `weights_backbone=ResNet50_Weights.DEFAULT` (ImageNet). The last layer is replaced with a 21-class head.
- **`train.py`** minimises the four Faster R-CNN losses (RPN objectness, RPN box, classifier, box) with SGD. Learning rate drops 10× every 3 epochs.
- **`predict.py`** loads the checkpoint in `eval()` mode and draws detections.