"""
train_kfold.py

Stratified Group K-Fold for the R(2+1)D-18 action classifier.
Designed for datasets where ALL files are augmented (no originals).

Why Stratified Group K-Fold?
-----------------------------
Your files:  aug_GaussianBlur_fight001_clip000.npy
             aug_GrayScale_fight001_clip000.npy
             aug_Brightness_fight001_clip000.npy

All three come from source video fight001.

- Plain K-Fold      : ignores class balance → skewed folds
- Stratified K-Fold : ignores the source group → leakage
                      (GaussianBlur_fight001 in train, GrayScale_fight001
                       in val = model already saw that video)
- Stratified GROUP K-Fold : keeps all augmentations of fight001 in
                             the SAME fold AND balances class ratios.
                             This is the correct choice.

Val / Test rule
---------------
Each val fold uses ONE augmentation type per source video to avoid
near-duplicate clips inflating the metric. The training fold keeps
all augmentation types for maximum diversity.

Usage:
    python train_kfold.py
    python train_kfold.py --folds 3   # use 3 folds if < 30 source videos/class
"""

import os
import sys
import time
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.cnn3d import build_model, unfreeze_backbone

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "/kaggle/working/data/processed"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR        = "/kaggle/working/runs"

N_FOLDS        = 5
CLIP_LEN       = 16
BATCH_SIZE     = 8
NUM_WORKERS    = 4       # set 0 on Windows if you hit DataLoader errors

PHASE1_EPOCHS  = 8
PHASE2_EPOCHS  = 22
PHASE1_LR      = 1e-3
PHASE2_LR      = 5e-5
UNFREEZE_LAYER = "layer3"
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTHING = 0.1
EARLY_STOP_PAT  = 6
SEED            = 42
# ───────────────────────────────────────────────────────────────────────────────

MEAN = [0.45, 0.406, 0.225]
STD  = [0.225, 0.224, 0.229]


# ─── Filename helpers ─────────────────────────────────────────────────────────

def get_base_name(stem: str) -> str:
    """aug_GaussianBlur_fight001_clip000 → fight001"""
    name = "_clip".join(stem.split("_clip")[:-1]) if "_clip" in stem else stem
    if name.startswith("aug_"):
        parts = name.split("_", 2)
        if len(parts) == 3:
            name = parts[2]
    return name


def get_aug_type(stem: str) -> str:
    """aug_GaussianBlur_fight001_clip000 → GaussianBlur"""
    if stem.startswith("aug_"):
        parts = stem.split("_", 2)
        if len(parts) >= 2:
            return parts[1]
    return "original"


# ─── Dataset ──────────────────────────────────────────────────────────────────

class AllClipsDataset:
    """
    Scans PROCESSED_DIR and builds flat lists needed for k-fold splitting.

    Exposes:
        samples    : list of (path_str, label_idx, aug_type, group_id)
        classes    : sorted class name list
        groups     : int array — same value = same source video (across classes)
        labels_arr : int array of label indices (for stratification)
    """

    def __init__(self, root: str):
        self.root    = Path(root)
        self.samples = []
        self.classes = []
        self.groups     = None
        self.labels_arr = None
        self._scan()

    def _scan(self):
        class_dirs   = sorted([d for d in self.root.iterdir() if d.is_dir()])
        self.classes = [d.name for d in class_dirs]

        source_to_id: dict[str, int] = {}
        next_id = 0

        for label_idx, class_dir in enumerate(class_dirs):
            for npy in sorted(class_dir.glob("*.npy")):
                base     = get_base_name(npy.stem)
                aug_type = get_aug_type(npy.stem)
                # Prefix class so fight001 ≠ normal001 in the group map
                key = f"{class_dir.name}__{base}"
                if key not in source_to_id:
                    source_to_id[key] = next_id
                    next_id += 1
                self.samples.append((str(npy), label_idx, aug_type, source_to_id[key]))

        self.groups     = np.array([s[3] for s in self.samples])
        self.labels_arr = np.array([s[1] for s in self.samples])

        print(f"[Dataset] {len(self.samples)} clips | "
              f"{len(self.classes)} classes | "
              f"{len(source_to_id)} source videos")
        print(f"[Dataset] Classes: {self.classes}")


class FoldDataset(Dataset):
    """
    Wraps a list of (path, label, aug_type) tuples for one fold.

    train=True  → keep ALL augmentation types, apply random augmentation
    train=False → keep only ONE aug type per source video (no near-dupes)
    """

    # Which aug type to use for val/test (deterministic — first alphabetically)
    EVAL_AUG_IDX = 0

    def __init__(self, parent: AllClipsDataset, indices: list[int],
                 clip_len: int, is_train: bool, all_aug_types: list[str]):
        self.parent       = parent
        self.clip_len     = clip_len
        self.is_train     = is_train
        self.eval_aug     = sorted(all_aug_types)[self.EVAL_AUG_IDX]

        if is_train:
            # Keep every clip assigned to this fold
            self.indices = indices
        else:
            # For eval: keep only ONE aug type per source video
            seen_sources: set[int] = set()
            filtered = []
            for i in indices:
                _, _, aug_type, group_id = parent.samples[i]
                if aug_type == self.eval_aug and group_id not in seen_sources:
                    seen_sources.add(group_id)
                    filtered.append(i)
            # Fallback: if that aug type missing, just take one clip per source
            if not filtered:
                seen_sources = set()
                for i in indices:
                    _, _, _, group_id = parent.samples[i]
                    if group_id not in seen_sources:
                        seen_sources.add(group_id)
                        filtered.append(i)
            self.indices = filtered

        print(f"  {'Train' if is_train else 'Val  '} fold: {len(self.indices)} clips")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        path, label, _, _ = self.parent.samples[self.indices[idx]]
        clip = np.load(path)             # (T, H, W, 3) uint8 RGB
        clip = self._ensure_len(clip)
        if self.is_train:
            clip = self._augment(clip)
        return self._to_tensor(clip), label

    def _ensure_len(self, clip):
        T = clip.shape[0]
        if T == self.clip_len:  return clip
        if T > self.clip_len:
            idx = np.linspace(0, T - 1, self.clip_len, dtype=int)
            return clip[idx]
        pad = np.stack([clip[-1]] * (self.clip_len - T))
        return np.concatenate([clip, pad])

    def _augment(self, clip):
        if random.random() < 0.5:
            clip = clip[:, :, ::-1, :].copy()
        if random.random() < 0.4:
            alpha = random.uniform(0.7, 1.3)
            beta  = random.randint(-20, 20)
            clip  = np.clip(clip.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return clip

    def _to_tensor(self, clip):
        t = clip.astype(np.float32) / 255.0
        t = torch.from_numpy(t).permute(3, 0, 1, 2)
        for c, (m, s) in enumerate(zip(MEAN, STD)):
            t[c] = (t[c] - m) / s
        return t


# ─── Training helpers ─────────────────────────────────────────────────────────

def get_class_weights(dataset: FoldDataset, num_classes: int, device):
    counts = Counter(dataset.parent.samples[i][1] for i in dataset.indices)
    total  = sum(counts.values())
    return torch.tensor(
        [total / (num_classes * max(counts[i], 1)) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0., 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out  = model(clips)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out  = model(clips)
            loss = criterion(out, labels)
        loss_sum += loss.item()
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), 100. * correct / total


def run_fold(fold, train_ds, val_ds, num_classes, device, ckpt_name):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold}")
    print(f"{'='*55}")

    model  = build_model(num_classes=num_classes, freeze_backbone=True).to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    cw     = get_class_weights(train_ds, num_classes, device)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True,
                              persistent_workers=(NUM_WORKERS > 0))

    best_val_acc     = 0.
    no_improve_count = 0

    for phase, epochs, lr, do_unfreeze in [
        (1, PHASE1_EPOCHS, PHASE1_LR, False),
        (2, PHASE2_EPOCHS, PHASE2_LR, True),
    ]:
        if do_unfreeze:
            unfreeze_backbone(model, unfreeze_from_layer=UNFREEZE_LAYER)
            no_improve_count = 0   # reset patience for phase 2

        criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
            vl_loss, vl_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(f"  Ph{phase} Ep{epoch:02d} ({time.time()-t0:.0f}s) | "
                  f"Train {tr_acc:.1f}% | Val {vl_acc:.1f}%")

            if vl_acc > best_val_acc:
                best_val_acc     = vl_acc
                no_improve_count = 0
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save({
                    "fold":        fold,
                    "model_state": model.state_dict(),
                    "val_acc":     vl_acc,
                    "classes":     train_ds.parent.classes,
                }, os.path.join(CHECKPOINT_DIR, ckpt_name))
                print(f"    ✓ Fold {fold} best: {vl_acc:.1f}%")
            else:
                no_improve_count += 1
                if no_improve_count >= EARLY_STOP_PAT:
                    print(f"  Early stopping (patience={EARLY_STOP_PAT})")
                    break

    return best_val_acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS,
                        help="Number of folds (default: 5). Use 3 if < 30 source videos/class.")
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    full_ds     = AllClipsDataset(PROCESSED_DIR)
    num_classes = len(full_ds.classes)

    # Collect all aug types present in the dataset
    all_aug_types = sorted({get_aug_type(Path(s[0]).stem) for s in full_ds.samples})
    print(f"Augmentation types found: {all_aug_types}")

    sgkf      = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=SEED)
    X         = np.arange(len(full_ds.samples))
    fold_accs = []

    for fold, (train_idx, val_idx) in enumerate(
        sgkf.split(X, full_ds.labels_arr, full_ds.groups), start=1
    ):
        print(f"\nFold {fold}/{args.folds}  "
              f"| raw train clips: {len(train_idx)}  raw val clips: {len(val_idx)}")

        train_ds = FoldDataset(full_ds, list(train_idx), CLIP_LEN,
                               is_train=True,  all_aug_types=all_aug_types)
        val_ds   = FoldDataset(full_ds, list(val_idx),   CLIP_LEN,
                               is_train=False, all_aug_types=all_aug_types)

        best = run_fold(
            fold, train_ds, val_ds, num_classes, device,
            ckpt_name=f"best_r2plus1d_fold{fold}.pth"
        )
        fold_accs.append(best)
        print(f"  Fold {fold} best val acc: {best:.1f}%")

    print(f"\n{'='*55}")
    print(f"  K-Fold Summary ({args.folds} folds)")
    for i, acc in enumerate(fold_accs, 1):
        print(f"    Fold {i}: {acc:.1f}%")
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  Mean ± Std : {mean_acc:.1f}% ± {std_acc:.1f}%")
    best_fold = int(np.argmax(fold_accs)) + 1
    print(f"  Best fold  : {best_fold}  ({max(fold_accs):.1f}%)")
    print(f"{'='*55}")
    print(f"\nUse checkpoint best_r2plus1d_fold{best_fold}.pth for final test evaluation.")
    print(f"If std > 5%, your dataset is too small or augmentations are too similar.")


if __name__ == "__main__":
    main()
