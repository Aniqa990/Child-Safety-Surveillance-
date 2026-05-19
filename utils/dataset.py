"""
dataset.py

PyTorch Dataset for .npy clips produced by preprocess_videos.py.
Clips are stored as (T, H, W, 3) uint8 RGB arrays.
Returns (3, T, H, W) float32 tensors normalised to Kinetics-400 stats.
"""

import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class ActionDataset(Dataset):
    """
    Loads segmented video clips from root/<class>/*.npy

    Expected clip format: (T, H, W, 3), uint8, RGB
    (produced by preprocess_videos.py)
    """

    # Kinetics-400 normalisation
    MEAN = [0.45, 0.406, 0.225]
    STD  = [0.225, 0.224, 0.229]

    def __init__(self, root: str, clip_len: int = 16, augment: bool = False):
        self.root     = Path(root)
        self.clip_len = clip_len
        self.augment  = augment
        self.samples  = []   # (path_str, label_idx)
        self.classes  = []
        self._scan()

    def _scan(self):
        class_dirs   = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]
        for label_idx, class_dir in enumerate(class_dirs):
            for npy in sorted(class_dir.glob("*.npy")):
                self.samples.append((str(npy), label_idx))
        print(f"[Dataset] {self.root.name}: {len(self.samples)} clips | "
              f"{len(self.classes)} classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        clip = np.load(path)              # (T, H, W, 3) uint8 RGB
        clip = self._ensure_len(clip)
        if self.augment:
            clip = self._augment(clip)
        return self._to_tensor(clip), label

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ensure_len(self, clip: np.ndarray) -> np.ndarray:
        T = clip.shape[0]
        if T == self.clip_len:
            return clip
        if T > self.clip_len:
            idx = np.linspace(0, T - 1, self.clip_len, dtype=int)
            return clip[idx]
        pad = np.stack([clip[-1]] * (self.clip_len - T))
        return np.concatenate([clip, pad], axis=0)

    def _augment(self, clip: np.ndarray) -> np.ndarray:
        """Light spatial/colour augmentations applied consistently per clip."""
        # Random horizontal flip
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()

        # Random brightness + contrast jitter
        if random.random() < 0.4:
            alpha = random.uniform(0.75, 1.25)   # contrast
            beta  = random.randint(-15, 15)       # brightness
            clip  = np.clip(
                clip.astype(np.float32) * alpha + beta, 0, 255
            ).astype(np.uint8)

        return clip

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 RGB → (3, T, H, W) float32, normalised."""
        t = clip.astype(np.float32) / 255.0
        t = torch.from_numpy(t).permute(3, 0, 1, 2)   # (3, T, H, W)
        for c, (m, s) in enumerate(zip(self.MEAN, self.STD)):
            t[c] = (t[c] - m) / s
        return t


def get_dataloaders(
    train_dir:   str,
    val_dir:     str,
    clip_len:    int = 16,
    batch_size:  int = 8,
    num_workers: int = 4,
):
    train_ds = ActionDataset(train_dir, clip_len=clip_len, augment=True)
    val_ds   = ActionDataset(val_dir,   clip_len=clip_len, augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    return train_loader, val_loader, train_ds.classes
