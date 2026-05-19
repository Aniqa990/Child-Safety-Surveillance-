"""
train.py

Training loop for R(2+1)D-18 on an augmentation-only dataset.

Features
--------
- Two-phase progressive unfreezing (head → layer3+)
- Label smoothing  → reduces overconfidence / overfitting
- AdamW + cosine LR schedule
- Class-weighted cross-entropy  → handles imbalance
- Early stopping  → stops when val plateaus
- Mixed-precision (autocast)  → faster GPU training
- Gradient clipping  → stable training

"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from collections import Counter

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.dataset import get_dataloaders
from models.cnn3d import build_model, unfreeze_backbone

# ─── CONFIG ────────────────────────────────────────────────────────────────────

TRAIN_DIR      = "/kaggle/working/data/train"
VAL_DIR        = "/kaggle/working/data/val"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR        = "/kaggle/working/runs"

CLIP_LEN    = 16
BATCH_SIZE  = 8
NUM_WORKERS = 4

# Phase 1 — classification head only
PHASE1_EPOCHS = 8
PHASE1_LR     = 1e-3

# Phase 2 — unfreeze from layer3 upwards
PHASE2_EPOCHS = 30
PHASE2_LR     = 5e-5

UNFREEZE_LAYER  = "layer3"
WEIGHT_DECAY    = 1e-4
LABEL_SMOOTHING = 0.1   # softens hard targets → better generalisation on small datasets
EARLY_STOP_PAT  = 8     # epochs without improvement before stopping a phase


def get_class_weights(dataloader, num_classes, device):
    counts = Counter()
    for _, labels in dataloader:
        counts.update(labels.tolist())
    total = sum(counts.values())
    return torch.tensor(
        [total / (num_classes * max(counts[i], 1)) for i in range(num_classes)],
        dtype=torch.float32
    ).to(device)


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    loss_sum, correct, total = 0., 0, 0
    for batch_idx, (clips, labels) in enumerate(loader):
        clips, labels = clips.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast(enabled=(device.type == "cuda")):
            out  = model(clips)
            loss = criterion(out, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        loss_sum += loss.item()
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx+1}/{len(loader)}] "
                  f"Loss {loss_sum/(batch_idx+1):.4f}  "
                  f"Acc {100.*correct/total:.1f}%")
    return loss_sum / len(loader), 100. * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        with autocast(enabled=(device.type == "cuda")):
            out  = model(clips)
            loss = criterion(out, labels)
        loss_sum += loss.item()
        correct  += out.argmax(1).eq(labels).sum().item()
        total    += labels.size(0)
    return loss_sum / len(loader), 100. * correct / total


def run_phase(phase, model, train_loader, val_loader, num_classes,
              device, writer, scaler, best_val_acc, epochs, lr,
              epoch_offset, ckpt_name):

    print(f"\n{'='*55}")
    print(f" PHASE {phase} — {epochs} epochs @ lr={lr}")
    print(f"{'='*55}")

    cw        = get_class_weights(train_loader, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=LABEL_SMOOTHING)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    no_improve = 0

    for epoch in range(1, epochs + 1):
        g  = epoch_offset + epoch
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"\nEpoch [{g}] ({time.time()-t0:.0f}s) | "
              f"Train {tr_acc:.1f}% loss {tr_loss:.4f} | "
              f"Val {vl_acc:.1f}% loss {vl_loss:.4f}")

        writer.add_scalars("Loss",     {"train": tr_loss, "val": vl_loss}, g)
        writer.add_scalars("Accuracy", {"train": tr_acc,  "val": vl_acc},  g)
        writer.add_scalar("LR", scheduler.get_last_lr()[0], g)

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            no_improve   = 0
            ckpt_path    = os.path.join(CHECKPOINT_DIR, ckpt_name)
            torch.save({
                "epoch":       g,
                "model_state": model.state_dict(),
                "val_acc":     vl_acc,
                "classes":     train_loader.dataset.classes,
            }, ckpt_path)
            print(f"  ✓ New best! Val {vl_acc:.1f}%  → saved {ckpt_name}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{EARLY_STOP_PAT})")
            if no_improve >= EARLY_STOP_PAT:
                print("  Early stopping triggered.")
                break

    return best_val_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    train_loader, val_loader, classes = get_dataloaders(
        TRAIN_DIR, VAL_DIR,
        clip_len=CLIP_LEN, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    num_classes = len(classes)
    print(f"Classes: {classes}")

    model  = build_model(num_classes=num_classes, freeze_backbone=True).to(device)
    writer = SummaryWriter(LOG_DIR)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # Phase 1: head only
    best_val_acc = run_phase(
        phase=1, model=model,
        train_loader=train_loader, val_loader=val_loader,
        num_classes=num_classes, device=device, writer=writer, scaler=scaler,
        best_val_acc=0., epochs=PHASE1_EPOCHS, lr=PHASE1_LR,
        epoch_offset=0, ckpt_name="best_r2plus1d.pth"
    )

    # Phase 2: unfreeze from layer3
    unfreeze_backbone(model, unfreeze_from_layer=UNFREEZE_LAYER)
    best_val_acc = run_phase(
        phase=2, model=model,
        train_loader=train_loader, val_loader=val_loader,
        num_classes=num_classes, device=device, writer=writer, scaler=scaler,
        best_val_acc=best_val_acc, epochs=PHASE2_EPOCHS, lr=PHASE2_LR,
        epoch_offset=PHASE1_EPOCHS, ckpt_name="best_r2plus1d.pth"
    )

    writer.close()
    print(f"\nTraining complete.  Best Val Accuracy: {best_val_acc:.1f}%")
    print(f"Checkpoint: {CHECKPOINT_DIR}/best_r2plus1d.pth")


if __name__ == "__main__":
    main()
