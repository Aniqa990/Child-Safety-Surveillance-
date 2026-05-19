"""
evaluate.py  —  Kaggle evaluation with fold checkpoints
=========================================================
Reads mp4 videos DIRECTLY from the Test-test dataset.
No preprocessing step needed — frames are extracted on the fly.

Two evaluation modes (set MODE below):
  "best_fold"  — single best-val-acc fold checkpoint. Fast.
  "ensemble"   — average softmax across ALL fold checkpoints. Recommended.

Paths (Kaggle)
--------------
  Checkpoints : /kaggle/input/r-2-1-d/checkpoints/best_r2plus1d_fold*.pt
                (Kaggle slugifies "r(2+1)d" notebook name → "r-2-1-d")
  Test videos : /kaggle/input/test-test/<folder>/*.mp4
  Output      : /kaggle/working/eval_results/

"""

import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

sys.path.append("/kaggle/working")
from models.cnn3d import build_model

# ─── CONFIG ───────────────────────────────────────────────────────────────────
# Kaggle slugifies the notebook name "r(2+1)d" → "r-2-1-d".
# If your sidebar shows a different slug, update this path.
CHECKPOINT_DIR  = Path("/kaggle/input/notebooks/aniqaazhar990/r-2-1-d/checkpoints")
CHECKPOINT_GLOB = "best_r2plus1d_fold*.pth"

TEST_DIR    = Path("/kaggle/input/datasets/aniqaazhar990/test-test")
OUTPUT_DIR  = Path("/kaggle/working/eval_results")

CLIP_LEN    = 16
FRAME_SIZE  = (112, 112)   # must match training preprocessing
BATCH_SIZE  = 4            # lower than training — mp4 decode is CPU-heavy
NUM_WORKERS = 2

# "best_fold" or "ensemble"
MODE = "best_fold"

# Mapping: test folder name  →  training class name
# Add / remove entries if your classes differ.
TEST_CLASS_MAP = {
    "climb" : "unsafeClimb",
    "fall"  : "fall",
    "fight" : "fight",
    "jump"  : "unsafeJump",
    "throw" : "unsafeThrow",
}
# ─────────────────────────────────────────────────────────────────────────────

MEAN = [0.45, 0.406, 0.225]
STD  = [0.225, 0.224, 0.229]
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}


# ── Video reading ─────────────────────────────────────────────────────────────

def read_video_as_clip(video_path: Path, clip_len: int, frame_size: tuple) -> np.ndarray:
    """
    Read an mp4 and return a single (T, H, W, 3) uint8 RGB clip.

    Strategy: uniformly sample `clip_len` frames across the full video
    so short and long videos are treated consistently — same strategy
    used by the dataset class during training.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < 1:
        cap.release()
        return None

    # Uniformly sample frame indices across the video
    if total_frames >= clip_len:
        indices = np.linspace(0, total_frames - 1, clip_len, dtype=int)
    else:
        # Video shorter than clip_len — use all frames then pad with last
        indices = list(range(total_frames)) + [total_frames - 1] * (clip_len - total_frames)
        indices = np.array(indices)

    frames = {}
    for idx in np.unique(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, frame_size)
            frames[int(idx)] = frame
    cap.release()

    clip = []
    for idx in indices:
        if int(idx) in frames:
            clip.append(frames[int(idx)])
        elif clip:
            clip.append(clip[-1])   # repeat last good frame
        else:
            clip.append(np.zeros((*frame_size[::-1], 3), dtype=np.uint8))

    return np.stack(clip, axis=0)   # (T, H, W, 3)


# ── Dataset ───────────────────────────────────────────────────────────────────

class VideoTestDataset(Dataset):
    """
    Reads mp4 videos from TEST_DIR/<folder>/*.mp4
    Maps folder names → training class names via TEST_CLASS_MAP.
    Filters to only include folders that have an entry in TEST_CLASS_MAP
    AND whose training class name exists in the checkpoint's class list.

    Returns: (3, T, H, W) float32 tensor, int label
    """

    def __init__(self, root: Path, clip_len: int,
                 frame_size: tuple, train_classes: list):
        self.root        = root
        self.clip_len    = clip_len
        self.frame_size  = frame_size
        self.classes     = train_classes          # ordered list from checkpoint
        self.class_to_idx = {c: i for i, c in enumerate(train_classes)}
        self.samples     = []   # (video_path, label_idx, train_class_name)

        self._scan()

    def _scan(self):
        skipped_folders = []
        for test_folder in sorted(self.root.iterdir()):
            if not test_folder.is_dir():
                continue

            folder_name = test_folder.name
            train_class = TEST_CLASS_MAP.get(folder_name)

            if train_class is None:
                skipped_folders.append(f"{folder_name} (not in TEST_CLASS_MAP)")
                continue
            if train_class not in self.class_to_idx:
                skipped_folders.append(
                    f"{folder_name} → '{train_class}' not in checkpoint classes"
                )
                continue

            label_idx = self.class_to_idx[train_class]
            videos    = [
                f for f in sorted(test_folder.iterdir())
                if f.suffix.lower() in VIDEO_EXTS
            ]
            for v in videos:
                self.samples.append((v, label_idx, train_class))

        # Print scan summary
        print(f"\n[TestDataset] Scan complete:")
        counts = {}
        for _, lbl, cls in self.samples:
            counts[cls] = counts.get(cls, 0) + 1
        for cls, n in sorted(counts.items()):
            print(f"  {cls:<18}: {n} videos")
        if skipped_folders:
            for s in skipped_folders:
                print(f"  [SKIP] {s}")
        print(f"  Total: {len(self.samples)} videos\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label, _ = self.samples[idx]
        clip = read_video_as_clip(video_path, self.clip_len, self.frame_size)
        if clip is None:
            print(f"  [WARN] Could not read {video_path.name}, using zeros.")
            clip = np.zeros((self.clip_len, *self.frame_size[::-1], 3), dtype=np.uint8)
        return self._to_tensor(clip), label

    def _to_tensor(self, clip: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 RGB → (3, T, H, W) float32 normalised."""
        t = clip.astype(np.float32) / 255.0
        t = torch.from_numpy(t).permute(3, 0, 1, 2)   # (3, T, H, W)
        for c, (m, s) in enumerate(zip(MEAN, STD)):
            t[c] = (t[c] - m) / s
        return t


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def find_checkpoints() -> list:
    ckpts = sorted(CHECKPOINT_DIR.glob(CHECKPOINT_GLOB))
    if not ckpts:
        raise FileNotFoundError(
            f"No checkpoints found matching:\n"
            f"  {CHECKPOINT_DIR / CHECKPOINT_GLOB}\n\n"
            f"Check the Kaggle sidebar to confirm the slug for your notebook.\n"
            f"Common slugs: 'r-2-1-d', 'r2-1-d'. Update CHECKPOINT_DIR above."
        )
    return ckpts


def pick_best_checkpoint(ckpt_paths: list) -> Path:
    best_path, best_acc = None, -1.0
    for p in ckpt_paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        acc  = ckpt.get("val_acc", 0.0)
        print(f"  {p.name}  val_acc={acc:.2f}%")
        if acc > best_acc:
            best_acc, best_path = acc, p
    print(f"\n  → Best: {best_path.name}  ({best_acc:.2f}%)\n")
    return best_path


def load_model(ckpt_path: Path, num_classes: int, device):
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_model(num_classes=num_classes, freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def get_probabilities(model, loader, device) -> np.ndarray:
    all_probs = []
    for clips, _ in loader:
        logits = model(clips.to(device))
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)   # (N, C)


def get_labels(loader) -> np.ndarray:
    return np.array([lbl for _, lbls in loader for lbl in lbls.tolist()])


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm, classes, output_path: Path, title: str):
    sz = max(7, len(classes))
    fig, ax = plt.subplots(figsize=(sz, sz - 1))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150)
    plt.close()
    print(f"  Saved → {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}")
    print(f"Mode        : {MODE}")
    print(f"Checkpoints : {CHECKPOINT_DIR}")
    print(f"Test data   : {TEST_DIR}")
    print(f"\nClass mapping (test folder → training class):")
    for k, v in TEST_CLASS_MAP.items():
        print(f"  {k:<10} → {v}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Checkpoints ───────────────────────────────────────────────────────
    ckpt_paths = find_checkpoints()
    print(f"\nFound {len(ckpt_paths)} checkpoint(s):")
    for p in ckpt_paths:
        print(f"  {p.name}")

    first_ckpt  = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
    classes     = first_ckpt["classes"]   # ordered list saved during training
    num_classes = len(classes)
    print(f"\nTraining classes ({num_classes}): {classes}")

    # ── Test loader ───────────────────────────────────────────────────────
    test_ds = VideoTestDataset(
        TEST_DIR, clip_len=CLIP_LEN,
        frame_size=FRAME_SIZE, train_classes=classes
    )
    if len(test_ds) == 0:
        raise RuntimeError(
            f"No test videos found.\n"
            f"Expected: {TEST_DIR}/<folder>/*.mp4\n"
            f"Folders present: {[d.name for d in TEST_DIR.iterdir() if d.is_dir()]}\n"
            f"TEST_CLASS_MAP keys: {list(TEST_CLASS_MAP.keys())}"
        )

    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    true_labels = get_labels(test_loader)

    # ── Inference ─────────────────────────────────────────────────────────
    if MODE == "best_fold":
        print("\n=== Mode: Best Fold ===")
        best_ckpt  = pick_best_checkpoint(ckpt_paths)
        model      = load_model(best_ckpt, num_classes, device)
        avg_probs  = get_probabilities(model, test_loader, device)
        mode_label = f"best fold ({best_ckpt.name})"

    elif MODE == "ensemble":
        print("\n=== Mode: Ensemble (all folds) ===")
        fold_probs = []
        for i, p in enumerate(ckpt_paths, 1):
            stored_acc = torch.load(p, map_location="cpu",
                                    weights_only=False).get("val_acc", 0.0)
            print(f"  Fold {i}: {p.name}  (val_acc={stored_acc:.1f}%)")
            model = load_model(p, num_classes, device)
            fold_probs.append(get_probabilities(model, test_loader, device))
            del model
            torch.cuda.empty_cache()

        avg_probs  = np.mean(fold_probs, axis=0)
        mode_label = f"ensemble of {len(ckpt_paths)} folds"

    else:
        raise ValueError(f"Unknown MODE='{MODE}'. Use 'best_fold' or 'ensemble'.")

    pred_labels = np.argmax(avg_probs, axis=1)

    # ── Results ───────────────────────────────────────────────────────────
    acc = 100.0 * np.mean(pred_labels == true_labels)

    print(f"\n{'='*55}")
    print(f"  {mode_label}")
    print(f"  Test videos : {len(true_labels)}")
    print(f"  Accuracy    : {acc:.2f}%")
    print(f"{'='*55}\n")

    report = classification_report(true_labels, pred_labels, target_names=classes)
    print(report)

    # Save text report
    report_path = OUTPUT_DIR / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(f"Mode: {mode_label}\n")
        f.write(f"Overall Accuracy: {acc:.2f}%\n\n")
        f.write(report)
    print(f"  Report → {report_path}")

    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    plot_confusion_matrix(
        cm, classes,
        output_path=OUTPUT_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix  {acc:.1f}%  ({mode_label})"
    )

    # Per-fold breakdown in ensemble mode
    if MODE == "ensemble":
        print(f"\n  Per-fold test accuracy:")
        for i, fp in enumerate(fold_probs, 1):
            fa = 100.0 * np.mean(np.argmax(fp, axis=1) == true_labels)
            print(f"    Fold {i}: {fa:.2f}%")
        print(f"    Ensemble : {acc:.2f}%  ← final number")


if __name__ == "__main__":
    evaluate()
