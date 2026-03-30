"""
utils/dataset.py

PyTorch Dataset that loads preprocessed .npy clips and returns
tensors in the format expected by 3D CNNs:  (C, T, H, W)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import random


class ActionDataset(Dataset):
    """
    Loads segmented video clips saved as .npy files.

    Expected structure:
        root/
            class_A/
                clip_001.npy   # shape (T, H, W, 3), uint8, 0-255
                clip_002.npy
            class_B/
                ...

    Returns:
        tensor : FloatTensor of shape (3, T, H, W), values in [0, 1]
        label  : int class index
    """

    # ImageNet / Kinetics mean and std (per channel)
    MEAN = [0.45, 0.406, 0.225]
    STD  = [0.225, 0.224, 0.229]

    def __init__(self, root: str, clip_len: int = 16, augment: bool = False):
        self.root      = Path(root)
        self.clip_len  = clip_len
        self.augment   = augment
        self.samples   = []   # list of (path, label_idx)
        self.classes   = []

        self._scan()

    def _scan(self):
        class_dirs = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        for label_idx, class_dir in enumerate(class_dirs):
            for npy_file in class_dir.glob("*.npy"):
                self.samples.append((str(npy_file), label_idx))

        print(f"[Dataset] {self.root.name}: {len(self.samples)} clips, "
              f"{len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = np.load(path)  # (T, H, W, 3), uint8

        clip = self._ensure_clip_len(clip)

        if self.augment:
            clip = self._augment(clip)

        tensor = self._to_tensor(clip)  # (3, T, H, W)
        return tensor, label

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ensure_clip_len(self, clip: np.ndarray) -> np.ndarray:
        """Uniformly sample or pad clip to self.clip_len frames."""
        T = clip.shape[0]
        if T == self.clip_len:
            return clip
        elif T > self.clip_len:
            indices = np.linspace(0, T - 1, self.clip_len, dtype=int)
            return clip[indices]
        else:
            # repeat last frame
            pad = np.stack([clip[-1]] * (self.clip_len - T), axis=0)
            return np.concatenate([clip, pad], axis=0)

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """Simple spatial augmentations applied consistently across frames."""
        # Random horizontal flip
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()

        # Random brightness jitter
        if random.random() < 0.3:
            factor = random.uniform(0.8, 1.2)
            clip = np.clip(clip.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        return clip

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """Convert (T, H, W, 3) uint8 numpy → (3, T, H, W) float32 tensor, normalised."""
        clip = clip.astype(np.float32) / 255.0          # (T, H, W, 3), [0,1]
        clip = torch.from_numpy(clip)                    # (T, H, W, 3)
        clip = clip.permute(3, 0, 1, 2)                  # (3, T, H, W)
        for c, (mean, std) in enumerate(zip(self.MEAN, self.STD)):
            clip[c] = (clip[c] - mean) / std
        return clip


def get_dataloaders(
    train_dir: str,
    val_dir:   str,
    clip_len:  int = 16,
    batch_size: int = 8,
    num_workers: int = 4,
):
    train_ds = ActionDataset(train_dir, clip_len=clip_len, augment=True)
    val_ds   = ActionDataset(val_dir,   clip_len=clip_len, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, train_ds.classes
