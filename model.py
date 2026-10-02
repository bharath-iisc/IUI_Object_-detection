"""Faster R-CNN with an ImageNet-pretrained ResNet-50 backbone.

The backbone is trained on ImageNet classification. Those weights already
encode edges, textures, and object parts, so the detector only has to learn
how to propose and classify boxes on VOC instead of learning vision from
scratch on ~17k images.
"""

from torchvision.models import ResNet50_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from voc import NUM_CLASSES


def get_model(num_classes: int = NUM_CLASSES, freeze_backbone: bool = False):
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=ResNet50_Weights.DEFAULT,
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False

    return model
