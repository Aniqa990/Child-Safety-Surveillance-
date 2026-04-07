"""
training/evaluate.py

Evaluates the saved best model on the test set and prints:
  - Per-class accuracy
  - Overall accuracy
  - Confusion matrix (saved as confusion_matrix.png)

Usage:
    python training/evaluate.py
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.dataset import ActionDataset
from models.cnn3d import build_model
from torch.utils.data import DataLoader

# ─── CONFIG ────────────────────────────────────────────────────────────────────
# TEST_DIR       = "data/test"
# Inside evaluate.py (for later!)
TEST_DIR = "G:/My Drive/Split_FYP_Data/test"
CHECKPOINT     = "G:/My Drive/Model_Checkpoints/best_r2plus1d.pth"
BATCH_SIZE     = 8
CLIP_LEN       = 16
NUM_WORKERS    = 4
OUTPUT_DIR     = "results"
# ───────────────────────────────────────────────────────────────────────────────


def plot_confusion_matrix(cm, classes, output_path):
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes, ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"[Eval] Confusion matrix saved to {output_path}")


@torch.no_grad()
def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Eval] Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(CHECKPOINT, map_location=device)
    classes = ckpt["classes"]
    num_classes = len(classes)
    print(f"[Eval] Classes: {classes}")

    # Build model and load weights
    model = build_model(num_classes=num_classes, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Test dataset
    test_ds = ActionDataset(TEST_DIR, clip_len=CLIP_LEN, augment=False)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )

    all_preds, all_labels = [], []

    for clips, labels in test_loader:
        clips = clips.to(device)
        outputs = model(clips)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.tolist())

    # ── Results ───────────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    acc = 100. * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n[Eval] Overall Accuracy: {acc:.2f}%")
    print(f"\n[Eval] Per-class Report:\n")
    print(classification_report(all_labels, all_preds, target_names=classes))

    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, classes, os.path.join(OUTPUT_DIR, "confusion_matrix.png"))


if __name__ == "__main__":
    evaluate()
