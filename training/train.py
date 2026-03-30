"""
training/train.py

Full training loop for the 3D CNN action classifier.

Features:
  - Transfer learning with progressive unfreezing
  - Learning rate scheduling (cosine annealing)
  - Model checkpointing (saves best val accuracy)
  - TensorBoard logging
  - Class imbalance handling via weighted loss

Usage:
    python training/train.py
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import get_dataloaders
from models.cnn3d import build_model, unfreeze_backbone

# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRAIN_DIR      = "data/train"
VAL_DIR        = "data/val"
CHECKPOINT_DIR = "checkpoints"
LOG_DIR        = "runs"

CLIP_LEN       = 16
BATCH_SIZE     = 8
NUM_WORKERS    = 0

# Phase 1: train head only
PHASE1_EPOCHS  = 5
PHASE1_LR      = 1e-3

# Phase 2: fine-tune from layer4
PHASE2_EPOCHS  = 15
PHASE2_LR      = 1e-4

UNFREEZE_LAYER = "layer4"   # 'layer4', 'layer3', 'layer2', or 'all'
WEIGHT_DECAY   = 1e-4
# ───────────────────────────────────────────────────────────────────────────────


def get_class_weights(dataloader, num_classes, device):
    """Compute inverse-frequency weights to handle class imbalance."""
    counts = Counter()
    for _, labels in dataloader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (num_classes * counts[i]) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)
    return weights


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch_idx, (clips, labels) in enumerate(loader):
        clips  = clips.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(clips)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.1f}%")

    return running_loss / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for clips, labels in loader:
        clips  = clips.to(device)
        labels = labels.to(device)
        outputs = model(clips)
        loss    = criterion(outputs, labels)

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total   += labels.size(0)

    return running_loss / len(loader), 100. * correct / total


def run_phase(
    phase, model, train_loader, val_loader, num_classes,
    device, writer, best_val_acc, epochs, lr
):
    print(f"\n{'='*50}")
    print(f" PHASE {phase} — {epochs} epochs @ lr={lr}")
    print(f"{'='*50}")

    class_weights = get_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    global_epoch_offset = 0 if phase == 1 else PHASE1_EPOCHS

    for epoch in range(1, epochs + 1):
        global_epoch = global_epoch_offset + epoch
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"\nEpoch [{global_epoch}] ({elapsed:.0f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.1f}%")

        # TensorBoard
        writer.add_scalars("Loss", {"train": train_loss, "val": val_loss}, global_epoch)
        writer.add_scalars("Accuracy", {"train": train_acc, "val": val_acc}, global_epoch)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], global_epoch)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch":        global_epoch,
                "model_state":  model.state_dict(),
                "val_acc":      val_acc,
                "classes":      train_loader.dataset.classes,
            }, ckpt_path)
            print(f"  ✅ New best! Val Acc: {val_acc:.1f}% — saved to {ckpt_path}")

    return best_val_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    train_loader, val_loader, classes = get_dataloaders(
        TRAIN_DIR, VAL_DIR,
        clip_len=CLIP_LEN,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    num_classes = len(classes)
    print(f"[Train] Classes: {classes}")

    model = build_model(num_classes=num_classes, freeze_backbone=True).to(device)
    writer = SummaryWriter(LOG_DIR)
    best_val_acc = 0.0

    # ── Phase 1: head only ────────────────────────────────────────────────────
    best_val_acc = run_phase(
        phase=1, model=model,
        train_loader=train_loader, val_loader=val_loader,
        num_classes=num_classes, device=device, writer=writer,
        best_val_acc=best_val_acc,
        epochs=PHASE1_EPOCHS, lr=PHASE1_LR
    )

    # ── Phase 2: unfreeze deeper layers ───────────────────────────────────────
    unfreeze_backbone(model, unfreeze_from_layer=UNFREEZE_LAYER)
    best_val_acc = run_phase(
        phase=2, model=model,
        train_loader=train_loader, val_loader=val_loader,
        num_classes=num_classes, device=device, writer=writer,
        best_val_acc=best_val_acc,
        epochs=PHASE2_EPOCHS, lr=PHASE2_LR
    )

    writer.close()
    print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"   Best model saved to: {CHECKPOINT_DIR}/best_model.pth")


if __name__ == "__main__":
    main()
