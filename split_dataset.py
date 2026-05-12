"""
split_dataset.py

Splits .npy clips into train / val / test WITHOUT data leakage.

Your dataset has ONLY augmented files — no originals:
    aug_GaussianBlur_fight001_clip000.npy
    aug_GrayScale_fight001_clip000.npy
    aug_Brightness_fight001_clip000.npy
    ... (6 augmentation types per source video)

Data leakage rule
-----------------
ALL augmentations of the same source video (fight001) must land
in the SAME split. If aug_GaussianBlur_fight001 goes to train and
aug_GrayScale_fight001 goes to val, the model has effectively seen
that video during training → inflated val accuracy, poor real-world results.

This script:
  1. Strips the aug_<Type>_ prefix to recover the source video name.
  2. Groups ALL augmentations of that source together.
  3. Assigns the ENTIRE group (all 6 aug types) to one split.
  4. Val and test each get ONE augmentation type per source video
     so evaluation clips are not near-duplicates of each other.

Usage:
    python split_dataset.py
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict

# ─── CONFIG ────────────────────────────────────────────────────────────────────
PROCESSED_DIR  = "/kaggle/working/data/processed"
SPLIT_ROOT     = "/kaggle/working/data"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR        = "/kaggle/working/runs"
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
TEST_RATIO    = 0.15
SEED          = 42
# ───────────────────────────────────────────────────────────────────────────────

assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-6


def get_base_name(stem: str) -> str:
    """
    Strip aug_<Type>_ prefix and _clipXXX suffix to get the source video name.

    aug_GaussianBlur_fight001_clip000  →  fight001
    aug_GrayScale_fight001_clip000     →  fight001
    """
    # Remove _clipXXX suffix
    name = "_clip".join(stem.split("_clip")[:-1]) if "_clip" in stem else stem
    # Remove aug_<Type>_ prefix
    if name.startswith("aug_"):
        parts = name.split("_", 2)   # ['aug', 'GaussianBlur', 'fight001']
        if len(parts) == 3:
            name = parts[2]
    return name


def get_aug_type(stem: str) -> str:
    """
    Extract augmentation type from filename stem.
    aug_GaussianBlur_fight001_clip000  →  GaussianBlur
    """
    if stem.startswith("aug_"):
        parts = stem.split("_", 2)
        if len(parts) >= 2:
            return parts[1]
    return "original"


def split_dataset():
    random.seed(SEED)
    processed = Path(PROCESSED_DIR)
    out_root  = Path(SPLIT_ROOT)

    for split in ["train", "val", "test"]:
        (out_root / split).mkdir(parents=True, exist_ok=True)

    class_dirs = sorted([d for d in processed.iterdir() if d.is_dir()])
    stats      = defaultdict(lambda: defaultdict(int))

    for class_dir in class_dirs:
        class_name = class_dir.name
        all_clips  = list(class_dir.glob("*.npy"))

        # ── Group clips by source video, then by aug type ──────────────────
        # source_groups[source_name][aug_type] = [clip_path, ...]
        source_groups = defaultdict(lambda: defaultdict(list))
        for clip in all_clips:
            base     = get_base_name(clip.stem)
            aug_type = get_aug_type(clip.stem)
            source_groups[base][aug_type].append(clip)

        source_names  = sorted(source_groups.keys())
        all_aug_types = sorted({
            aug_type
            for grp in source_groups.values()
            for aug_type in grp
        })
        random.shuffle(source_names)
        n = len(source_names)

        print(f"\n[Split] {class_name}: {n} source videos | "
              f"aug types: {all_aug_types} | "
              f"{len(all_clips)} total clips")

        # ── Assign source videos to splits ─────────────────────────────────
        if n < 3:
            print(f"  [WARN] Only {n} source videos — cannot make a 3-way split. "
                  f"All clips → train.")
            assign = {"train": source_names, "val": [], "test": []}
        else:
            n_train = max(1, round(n * TRAIN_RATIO))
            n_val   = max(1, round(n * VAL_RATIO))
            n_test  = n - n_train - n_val
            if n_test < 1:
                n_test  = 1
                n_train = n - n_val - n_test
            if n_val < 1:
                n_val   = 1
                n_train = n - n_val - n_test
            while n_train + n_val + n_test > n:
                if n_train > 1: n_train -= 1
                elif n_val > 1: n_val   -= 1
                else:           n_test  -= 1
            while n_train + n_val + n_test < n:
                n_train += 1

            assign = {
                "train": source_names[:n_train],
                "val":   source_names[n_train : n_train + n_val],
                "test":  source_names[n_train + n_val :],
            }

        print(f"  → {len(assign['train'])} train | "
              f"{len(assign['val'])} val | "
              f"{len(assign['test'])} test  source videos")

        # ── Pick which aug types go to val and test ─────────────────────────
        # Use different aug types for val vs test so they don't overlap.
        # Any aug type is equally valid — none of these source videos were
        # seen during training at all.
        val_aug_type  = all_aug_types[0] if len(all_aug_types) >= 1 else None
        test_aug_type = all_aug_types[1] if len(all_aug_types) >= 2 else all_aug_types[0]

        # ── Copy clips ────────────────────────────────────────────────────
        for split_name, src_list in assign.items():
            dst_dir = out_root / split_name / class_name
            dst_dir.mkdir(parents=True, exist_ok=True)

            for src in src_list:
                aug_dict = source_groups[src]

                if split_name == "train":
                    # ALL augmentation types → more training diversity
                    for clips_list in aug_dict.values():
                        for clip in clips_list:
                            shutil.copy(str(clip), str(dst_dir / clip.name))
                            stats[class_name]["train"] += 1

                elif split_name == "val":
                    # ONE aug type only → no near-duplicate clips in val
                    target = val_aug_type if val_aug_type in aug_dict else next(iter(aug_dict))
                    for clip in aug_dict[target]:
                        shutil.copy(str(clip), str(dst_dir / clip.name))
                        stats[class_name]["val"] += 1

                elif split_name == "test":
                    # A DIFFERENT aug type from val
                    target = test_aug_type if test_aug_type in aug_dict else next(iter(aug_dict))
                    for clip in aug_dict[target]:
                        shutil.copy(str(clip), str(dst_dir / clip.name))
                        stats[class_name]["test"] += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  Dataset split summary")
    print(f"  {'Class':<22} {'Train':>7} {'Val':>7} {'Test':>7} {'Total':>8}")
    print(f"  {'─'*55}")
    for cls in sorted(stats.keys()):
        tr = stats[cls]["train"]
        va = stats[cls]["val"]
        te = stats[cls]["test"]
        print(f"  {cls:<22} {tr:>7} {va:>7} {te:>7} {tr+va+te:>8}")
    print(f"\nDone.  Files written to {SPLIT_ROOT}/{{train,val,test}}/")
    print("\nNotes:")
    print("  Train → ALL augmentation types for its assigned source videos.")
    print("  Val   → ONE aug type per source video (avoids near-duplicates).")
    print("  Test  → a DIFFERENT aug type per source video.")
    print("  No source video appears in more than one split → zero leakage.")


if __name__ == "__main__":
    split_dataset()
