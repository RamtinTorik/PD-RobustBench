"""Model builders: three ImageNet-pretrained CNN backbones with a light,
uniformly-structured classification head. Backbones stay frozen (ImageNet
features only); only the head is trained, per the project design.

Feature dims after global average pooling:
    resnet50        -> 2048
    vgg16           -> 512  (GAP over the final conv feature map; the heavy
                             25088->4096 classifier is intentionally replaced)
    efficientnet_b0 -> 1280

The head is: Flatten -> Dropout(0.3) -> Linear(feat,64) -> ReLU -> Linear(64,1)
Output is a single logit (BCEWithLogitsLoss during training).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from config import MODELS

FEAT_DIMS = {"resnet50": 2048, "vgg16": 512, "efficientnet_b0": 1280,
             "mobilenet_v3_large": 960}

# last convolutional feature-map module per backbone, used by Grad-CAM
GRADCAM_LAYER = {
    "resnet50": ("backbone", 7),          # layer4 (final residual stage)
    "vgg16": ("backbone", (0, 30)),       # features[30] = final max-pooled conv map
    "efficientnet_b0": ("backbone", (0, 8)),  # features[8] = final MBConv stage
    "mobilenet_v3_large": ("backbone", (0, -1)),  # features[-1] = final 960-ch conv stage
}


def build_feature_extractor(name: str) -> tuple:
    """Return (frozen backbone module producing (B, feat_dim, 1, 1), feat_dim)."""
    if name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(*list(base.children())[:-2], nn.AdaptiveAvgPool2d(1))
    elif name == "vgg16":
        base = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
    elif name == "efficientnet_b0":
        base = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
    elif name == "mobilenet_v3_large":
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        backbone = nn.Sequential(base.features, nn.AdaptiveAvgPool2d(1))
    else:
        raise ValueError(f"Unknown model: {name}; choose from {MODELS}")
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone, FEAT_DIMS[name]


def build_head(feat_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(feat_dim, 64),
        nn.ReLU(inplace=True),
        nn.Linear(64, 1),
    )


class PDClassifier(nn.Module):
    """Backbone (frozen, always eval mode) + trainable head."""

    def __init__(self, name: str, head_state: dict = None):
        super().__init__()
        self.name = name
        self.backbone, feat_dim = build_feature_extractor(name)
        self.backbone.eval()
        self.head = build_head(feat_dim)
        if head_state is not None:
            self.head.load_state_dict(head_state)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()  # keep frozen BN/dropout of the backbone in eval
        return self

    def forward(self, x):
        feats = self.backbone(x).flatten(1)
        return self.head(feats)


def gradcam_target_module(model: PDClassifier) -> nn.Module:
    """Resolve the Grad-CAM target conv module for a given architecture."""
    spec = GRADCAM_LAYER[model.name]
    mod = getattr(model, spec[0])           # e.g. model.backbone
    rest = spec[1]
    if isinstance(rest, int):
        return mod[rest]
    a, b = rest
    return mod[a][b]


def last_stage_module(model: PDClassifier) -> nn.Module:
    """The final backbone stage that is unfrozen for light fine-tuning:
    ResNet50 -> layer4, VGG16 -> conv block 5 (features[24:]),
    EfficientNet-B0 -> the last three feature blocks (features[6:]),
    MobileNetV3-Large -> the last three feature blocks (features[-3:])."""
    if model.name == "resnet50":
        return model.backbone[7]
    if model.name == "vgg16":
        return model.backbone[0][24:]
    if model.name == "efficientnet_b0":
        return model.backbone[0][6:]
    if model.name == "mobilenet_v3_large":
        return model.backbone[0][-3:]
    raise ValueError(model.name)


def ckpt_name(dtype: str, model: str, seed: int = 42) -> str:
    """Checkpoint file name; the default seed keeps the canonical name."""
    suffix = "" if seed == 42 else f"__s{seed}"
    return f"{dtype}__{model}{suffix}.pt"


def load_model(name: str, ckpt_path) -> PDClassifier:
    """Build a model and load trained weights (head + optionally the
    fine-tuned last backbone stage) from disk."""
    state = torch.load(ckpt_path, map_location="cpu")
    model = PDClassifier(name, head_state=state["head"])
    if "last_stage" in state and state["last_stage"]:
        last_stage_module(model).load_state_dict(state["last_stage"])
    model.eval()
    return model


def load_threshold(ckpt_path) -> float:
    """Decision threshold selected on the validation split at train time."""
    state = torch.load(ckpt_path, map_location="cpu")
    return float(state.get("threshold", 0.5))
