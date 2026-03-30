"""
models/cnn3d.py

Loads a pretrained R(2+1)D-18 from torchvision and replaces the
classification head with one matching your number of action classes.

R(2+1)D decomposes 3D convolutions into:
  - a 2D spatial convolution
  - a 1D temporal convolution
This gives better accuracy than pure C3D with fewer parameters.

Pretrained weights: Kinetics-400 (human action recognition dataset)
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


def build_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Build fine-tunable R(2+1)D-18 model.

    Args:
        num_classes     : number of action classes in your dataset
        freeze_backbone : if True, only the final FC layer is trained initially.
                          Set to False (or call unfreeze_backbone()) for full fine-tuning.

    Returns:
        model : nn.Module ready for training
    """
    # Load pretrained weights (downloads ~120 MB on first run)
    weights = R2Plus1D_18_Weights.KINETICS400_V1
    model   = r2plus1d_18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    print(f"[Model] R(2+1)D-18 loaded — {num_classes} classes")
    print(f"[Model] Backbone {'frozen' if freeze_backbone else 'unfrozen'}")
    _print_param_count(model)

    return model


def unfreeze_backbone(model: nn.Module, unfreeze_from_layer: str = "layer4"):
    """
    Progressively unfreeze layers for fine-tuning.

    Recommended schedule:
      Epoch 1-5  : freeze_backbone=True  (train head only)
      Epoch 6+   : unfreeze layer4 (and optionally layer3)

    Args:
        unfreeze_from_layer : 'layer4', 'layer3', 'layer2', or 'all'
    """
    layers_to_unfreeze = {
        "layer4": ["layer4", "fc"],
        "layer3": ["layer3", "layer4", "fc"],
        "layer2": ["layer2", "layer3", "layer4", "fc"],
        "all"   : None,  # unfreeze everything
    }

    target = layers_to_unfreeze.get(unfreeze_from_layer)

    for name, param in model.named_parameters():
        if target is None:
            param.requires_grad = True
        else:
            param.requires_grad = any(name.startswith(l) for l in target)

    print(f"[Model] Unfrozen from: {unfreeze_from_layer}")
    _print_param_count(model)


def _print_param_count(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {trainable:,} / {total:,}")
