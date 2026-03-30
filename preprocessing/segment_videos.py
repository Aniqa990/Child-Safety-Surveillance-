"""
preprocessing/segment_videos.py

Segments videos using MOG2 background subtraction and saves
cleaned clips as .npy arrays ready for 3D CNN training.

Expected dataset structure:
    data/raw/
        class_A/
            video1.mp4
            video2.mp4
        class_B/
            video1.mp4
            ...

Output structure:
    data/processed/
        class_A/
            video1.npy   # shape: (T, H, W, 3)
            video2.npy
        class_B/
            ...
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import imageio  # for GIF saving

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DIR       = "data/raw"
OUTPUT_DIR    = "data/processed"
CLIP_LEN      = 16          # number of frames per clip
FRAME_SIZE    = (112, 112)  # resize each frame to this (H, W)
WARMUP_FRAMES = 50          # frames used to initialize background model
MOG2_HISTORY  = 300         # how many frames MOG2 remembers
MOG2_THRESH   = 40          # sensitivity — lower = more sensitive
OVERLAP       = 0           # frame overlap between clips (0 = no overlap)
# ───────────────────────────────────────────────────────────────────────────────


def build_bg_subtractor():
    return cv2.createBackgroundSubtractorMOG2(
        history=MOG2_HISTORY,
        varThreshold=MOG2_THRESH,
        detectShadows=False
    )


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """Remove noise and fill holes in the foreground mask."""
    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open  = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)  # fill holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)   # remove speckles
    return mask


def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest foreground blob (the child)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    clean = np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
    return clean


def segment_frame(frame: np.ndarray, bg_sub) -> np.ndarray:
    """Apply MOG2 + cleanup to a single frame. Returns masked RGB frame."""
    mask = bg_sub.apply(frame)
    mask = clean_mask(mask)
    mask = keep_largest_contour(mask)
    segmented = cv2.bitwise_and(frame, frame, mask=mask)
    resized = cv2.resize(segmented, FRAME_SIZE)
    return resized


def extract_clips(frames: list) -> list:
    """
    Split a list of frames into fixed-length clips.
    Clips that are too short are padded by repeating the last frame.
    """
    clips = []
    step = CLIP_LEN - OVERLAP
    for start in range(0, len(frames), step):
        clip = frames[start : start + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            # pad by repeating last frame
            clip += [clip[-1]] * (CLIP_LEN - len(clip))
        if len(clip) == CLIP_LEN:
            clips.append(np.stack(clip, axis=0))  # (T, H, W, 3)
    return clips


def save_clip_as_gif(clip: np.ndarray, output_path: str, fps: int = 8):
    """Save a (T, H, W, 3) clip as GIF for visualization."""
    frames = [clip[i] for i in range(clip.shape[0])]
    imageio.mimsave(output_path, frames, fps=fps)
    print(f"  [GIF] Saved {output_path}")


def process_video(video_path: str, save_gif: bool = False, out_root: Path = None) -> list:
    """
    Full pipeline for one video:
      1. Warm up MOG2 on first WARMUP_FRAMES frames
      2. Segment remaining frames
      3. Split into fixed-length clips
    Returns list of np.ndarray clips, each shape (T, H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping.")
        return []

    bg_sub = build_bg_subtractor()

    # ── Warm-up pass ──────────────────────────────────────────────────────────
    for _ in range(WARMUP_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        bg_sub.apply(frame)

    # ── Segmentation pass ─────────────────────────────────────────────────────
    segmented_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        seg_frame = segment_frame(frame, bg_sub)
        segmented_frames.append(seg_frame)

    cap.release()

    if len(segmented_frames) < CLIP_LEN:
        print(f"  [WARN] {video_path} is too short ({len(segmented_frames)} frames), skipping.")
        return []

    clips = extract_clips(segmented_frames)

    # Save first clip as GIF for visualization
    if save_gif and clips and out_root:
        video_name = Path(video_path).stem
        gif_path = out_root / f"{video_name}_clip0.gif"
        save_clip_as_gif(clips[0], str(gif_path))

    return clips


def process_dataset():
    """Walk RAW_DIR, process every video, and save clips to OUTPUT_DIR."""
    raw_root = Path(RAW_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    class_dirs = [d for d in raw_root.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"[ERROR] No class subdirectories found in {RAW_DIR}")
        return

    total_clips = 0

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        out_class_dir = out_root / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        videos = [f for f in class_dir.iterdir() if f.suffix.lower() in video_exts]
        print(f"\n[CLASS] {class_name} — {len(videos)} videos")

        # Save GIF for first video of each class
        save_gif_for_first = True

        for video_path in tqdm(videos, desc=f"  {class_name}"):
            clips = process_video(str(video_path), save_gif=save_gif_for_first, out_root=out_root)
            save_gif_for_first = False  # Only save for first video

            for i, clip in enumerate(clips):
                out_path = out_class_dir / f"{video_path.stem}_clip{i:03d}.npy"
                np.save(str(out_path), clip)
                total_clips += 1

    print(f"\n✅ Done. Total clips saved: {total_clips}")
    print(f"   Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    process_dataset()
