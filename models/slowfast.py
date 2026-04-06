"""
models/slowfast.py

SlowFast network for action recognition.
Pretrained on Kinetics-400 via pytorchvideo (Facebook Research).

SlowFast uses two pathways:
  - Slow pathway : low frame rate, captures spatial semantics (WHAT is happening)
  - Fast pathway : high frame rate, captures motion dynamics (HOW FAST it's happening)

This is particularly good for your use case because:
  - Distinguishes similar actions: Climb vs Unsafe_Climb, Throw vs Unsafe_Throw
  - Captures both fine detail and fast movement simultaneously
  - Pretrained on Kinetics-400 which includes fighting, climbing, jumping

Input format is different from R(2+1)D — SlowFast expects a LIST of two tensors:
  [slow_tensor, fast_tensor]
  slow: (B, 3, T//alpha, H, W)  — e.g. (B, 3, 8,  224, 224) with alpha=4
  fast: (B, 3, T,        H, W)  — e.g. (B, 3, 32, 224, 224)

Usage:
    from models.slowfast import build_slowfast_model, prepare_slowfast_input
    model = build_slowfast_model(num_classes=7)
"""

import torch
import torch.nn as nn


# SlowFast alpha — ratio of fast to slow frame rate
ALPHA = 4   # fast pathway has 4x more frames than slow


def build_slowfast_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:
    """
    Load pretrained SlowFast R50 from pytorchvideo and replace
    the classification head with one matching your classes.

    Args:
        num_classes     : number of action classes in your dataset
        freeze_backbone : if True, only head is trained initially

    Returns:
        model : nn.Module ready for training
    """
    # Load pretrained SlowFast from torch hub
    print("[Model] Loading SlowFast R50 (pretrained Kinetics-400)...")
    print("[Model] This may take a moment on first run (~160 MB download)...")

    model = torch.hub.load(
        "facebookresearch/pytorchvideo",
        "slowfast_r50",
        pretrained=True
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace classification head
    # SlowFast head is model.blocks[-1].proj
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes)
    )

    print(f"[Model] SlowFast R50 loaded — {num_classes} classes")
    print(f"[Model] Backbone {'frozen' if freeze_backbone else 'unfrozen'}")
    _print_param_count(model)

    return model


def unfreeze_slowfast(model: nn.Module, unfreeze_from: str = "res5"):
    """
    Progressively unfreeze SlowFast layers.

    SlowFast layer names:
      blocks.0 → stem (slow)
      blocks.1 → stem (fast)
      blocks.2 → res2
      blocks.3 → res3
      blocks.4 → res4
      blocks.5 → res5   ← unfreeze from here first
      blocks.6 → head

    Args:
        unfreeze_from : 'res5', 'res4', 'res3', or 'all'
    """
    layer_map = {
        "res5": ["blocks.5", "blocks.6"],
        "res4": ["blocks.4", "blocks.5", "blocks.6"],
        "res3": ["blocks.3", "blocks.4", "blocks.5", "blocks.6"],
        "all" : None,
    }

    target = layer_map.get(unfreeze_from)

    for name, param in model.named_parameters():
        if target is None:
            param.requires_grad = True
        else:
            param.requires_grad = any(name.startswith(l) for l in target)

    print(f"[Model] SlowFast unfrozen from: {unfreeze_from}")
    _print_param_count(model)


def prepare_slowfast_input(clips: torch.Tensor, alpha: int = ALPHA):
    """
    Convert a standard clip tensor into SlowFast dual-pathway input.

    SlowFast requires TWO tensors as a list:
      - slow : subsampled frames  (every alpha-th frame)
      - fast : all frames

    Args:
        clips : (B, 3, T, H, W) standard clip tensor
        alpha : slow/fast ratio (default 4)

    Returns:
        [slow_tensor, fast_tensor]
    """
    # Fast pathway — all frames
    fast = clips  # (B, 3, T, H, W)

    # Slow pathway — every alpha-th frame
    slow = clips[:, :, ::alpha, :, :]  # (B, 3, T//alpha, H, W)

    return [slow, fast]


def _print_param_count(model: nn.Module):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Trainable params: {trainable:,} / {total:,}")