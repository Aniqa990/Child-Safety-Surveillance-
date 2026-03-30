"""
preprocessing/split_dataset.py

Splits processed .npy clips into train / val / test sets.

Run this AFTER segment_videos.py.

Usage:
    python preprocessing/split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR = "data/processed"
SPLIT_ROOT    = "data"
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
SEED          = 42
# ───────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6, "Ratios must sum to 1"


def split_dataset():
    random.seed(SEED)
    processed = Path(PROCESSED_DIR)
    splits = {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO}

    # Create output dirs
    for split in splits:
        (Path(SPLIT_ROOT) / split).mkdir(parents=True, exist_ok=True)

    class_dirs = [d for d in processed.iterdir() if d.is_dir()]
    stats = defaultdict(lambda: defaultdict(int))

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        clips = list(class_dir.glob("*.npy"))

        # Group clips by source video (assuming naming: video_name_clip_idx.npy)
        video_groups = defaultdict(list)
        for clip in clips:
            # Extract video name (everything before last underscore)
            video_name = "_".join(clip.stem.split("_")[:-1])
            video_groups[video_name].append(clip)

        # Get list of videos and shuffle
        videos = list(video_groups.keys())
        random.shuffle(videos)

        n = len(videos)
        n_train = int(n * TRAIN_RATIO)
        n_val   = int(n * VAL_RATIO)

        split_videos = {
            "train": videos[:n_train],
            "val":   videos[n_train:n_train + n_val],
            "test":  videos[n_train + n_val:],
        }

        # Assign all clips from each video to the same split
        for split, video_list in split_videos.items():
            dst_dir = Path(SPLIT_ROOT) / split / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for video in video_list:
                for clip in video_groups[video]:
                    shutil.copy(str(clip), str(dst_dir / clip.name))
                    stats[class_name][split] += 1

    # Print summary
    print("\n📊 Dataset split summary:")
    print(f"  {'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6} {'Total':>7}")
    print("  " + "-" * 45)
    for cls in sorted(stats.keys()):
        tr = stats[cls]["train"]
        va = stats[cls]["val"]
        te = stats[cls]["test"]
        print(f"  {cls:<20} {tr:>6} {va:>6} {te:>6} {tr+va+te:>7}")
    print(f"\n✅ Done. Files saved to {SPLIT_ROOT}/{{train,val,test}}/")


if __name__ == "__main__":
    split_dataset()
