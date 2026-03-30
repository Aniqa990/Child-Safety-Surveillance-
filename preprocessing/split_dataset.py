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

        if len(videos) <= 1:
            # If class has one or zero source videos, do clip-level split.
            print(f"[Split] {class_name}: using clip-level split (only {len(videos)} source video)")
            random.shuffle(clips)
            n = len(clips)
            n_train = int(round(n * TRAIN_RATIO))
            n_val = int(round(n * VAL_RATIO))
            n_test = n - n_train - n_val

            if n_val == 0 and n >= 2:
                n_val = 1
            if n_test == 0 and n >= 3:
                n_test = 1

            # adjust to keep sum = n
            if n_train + n_val + n_test > n:
                overflow = n_train + n_val + n_test - n
                if n_test > 0:
                    n_test -= overflow
                elif n_val > 0:
                    n_val -= overflow
                else:
                    n_train -= overflow

            if n_train + n_val + n_test < n:
                n_train += n - (n_train + n_val + n_test)

            split_clips = {
                "train": clips[:n_train],
                "val": clips[n_train:n_train + n_val],
                "test": clips[n_train + n_val:],
            }

            for split, split_list in split_clips.items():
                dst_dir = Path(SPLIT_ROOT) / split / class_name
                dst_dir.mkdir(parents=True, exist_ok=True)
                for clip in split_list:
                    shutil.copy(str(clip), str(dst_dir / clip.name))
                    stats[class_name][split] += 1

            continue

        n = len(videos)
        # Compute target counts
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        n_test = n - n_train - n_val

        if n == 2:
            # 2-source-video case: keep small but non-empty validation
            n_train = 1
            n_val = 1
            n_test = 0
        elif n >= 3:
            # Ensure each split gets at least one source video
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)

            # Fix total if needed
            while n_train + n_val + n_test > n:
                if n_test > 1:
                    n_test -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    n_train -= 1
            while n_train + n_val + n_test < n:
                n_train += 1
        else:
            # n==0 or n==1 should be handled above, but keep safe fallback
            n_train = n
            n_val = 0
            n_test = 0

        print(f"[Split] {class_name}: using video-level split ({n} source videos => {n_train} train, {n_val} val, {n_test} test)")
        split_videos = {
            "train": videos[:n_train],
            "val": videos[n_train:n_train + n_val],
            "test": videos[n_train + n_val:],
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
