"""
check.py

Data leakage checker for augmentation-only datasets.

Correctly handles filenames like:
    aug_GaussianBlur_fight001_clip000.npy
    aug_GrayScale_fight001_clip000.npy

Both map to source video 'fight001'. This script checks that no
source video appears in more than one split.

Also warns if val/test contain multiple aug types of the same source
(which inflates metrics by including near-duplicate clips in evaluation).

Usage:
    python check.py
"""

from pathlib import Path
from collections import defaultdict

SPLIT_ROOT = "data"


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


def check_leakage():
    splits = ["train", "val", "test"]

    # split → class → set of source names
    source_sets  = {s: defaultdict(set) for s in splits}
    # split → class → source → set of aug types
    aug_presence = {s: defaultdict(lambda: defaultdict(set)) for s in splits}
    clip_counts  = {s: 0 for s in splits}

    for split in splits:
        split_dir = Path(SPLIT_ROOT) / split
        if not split_dir.exists():
            continue
        for npy in split_dir.rglob("*.npy"):
            cls      = npy.parent.name
            base     = get_base_name(npy.stem)
            aug_type = get_aug_type(npy.stem)
            source_sets[split][cls].add(base)
            aug_presence[split][cls][base].add(aug_type)
            clip_counts[split] += 1

    # ── Summary ────────────────────────────────────────────────────────────
    print("Dataset summary")
    print("─" * 55)
    all_classes = sorted({c for s in splits for c in source_sets[s]})
    for split in splits:
        n_sources = sum(len(v) for v in source_sets[split].values())
        print(f"  {split:<6}: {clip_counts[split]:>6} clips | {n_sources:>4} source videos")

    # ── Leakage check ──────────────────────────────────────────────────────
    print()
    leakage_found = False
    for cls in all_classes:
        tv = source_sets["train"][cls] & source_sets["val"][cls]
        tt = source_sets["train"][cls] & source_sets["test"][cls]
        vt = source_sets["val"][cls]   & source_sets["test"][cls]
        if tv or tt or vt:
            leakage_found = True
            print(f"[LEAK] Class '{cls}':")
            if tv: print(f"  train ∩ val  ({len(tv)}): {sorted(tv)[:5]}")
            if tt: print(f"  train ∩ test ({len(tt)}): {sorted(tt)[:5]}")
            if vt: print(f"  val ∩ test   ({len(vt)}): {sorted(vt)[:5]}")

    if not leakage_found:
        print("✓ No source video leakage detected across splits.")

    # ── Near-duplicate check in val/test ───────────────────────────────────
    print()
    near_dup_found = False
    for split in ["val", "test"]:
        for cls in all_classes:
            multi_aug = {
                src: aug_types
                for src, aug_types in aug_presence[split][cls].items()
                if len(aug_types) > 1
            }
            if multi_aug:
                near_dup_found = True
                print(f"[WARN] {split}/{cls}: {len(multi_aug)} source videos "
                      f"have multiple aug types (near-duplicates in eval):")
                for src, augs in list(multi_aug.items())[:3]:
                    print(f"  {src}: {sorted(augs)}")

    if not near_dup_found:
        print("✓ Val and test sets each use one aug type per source video "
              "(no near-duplicate clips in evaluation).")

    # ── Aug type distribution ──────────────────────────────────────────────
    print()
    print("Augmentation type distribution per split:")
    for split in splits:
        aug_type_counts: dict[str, int] = defaultdict(int)
        split_dir = Path(SPLIT_ROOT) / split
        if not split_dir.exists():
            continue
        for npy in split_dir.rglob("*.npy"):
            aug_type_counts[get_aug_type(npy.stem)] += 1
        print(f"  {split}: {dict(sorted(aug_type_counts.items()))}")


if __name__ == "__main__":
    check_leakage()
