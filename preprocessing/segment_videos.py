# """
# preprocessing/segment_videos.py

# Segments videos using YOLO (default) or MOG2 background subtraction
# and saves cleaned clips as .npy arrays ready for 3D CNN training.

# Expected dataset structure:
#     data/raw/
#         class_A/
#             video1.mp4
#             video2.mp4
#         class_B/
#             video1.mp4
#             ...

# Output structure:
#     data/processed/
#         class_A/
#             video1_clip000.npy   # shape: (T, H, W, 3)
#             video1_clip001.npy
#         class_B/
#             ...
# """

# import cv2
# import numpy as np
# import os
# from pathlib import Path
# from tqdm import tqdm
# from ultralytics import YOLO

# # ─── CONFIG ────────────────────────────────────────────────────────────────────
# RAW_DIR       = "data/raw"
# OUTPUT_DIR    = "data/processed"
# CLIP_LEN      = 16            # number of frames per clip
# FRAME_SIZE    = (112, 112)    # resize each frame to (W, H) for 3D CNN
# WARMUP_FRAMES = 50            # frames used to initialize MOG2 (MOG2 only)
# MOG2_HISTORY  = 300           # how many frames MOG2 remembers
# MOG2_THRESH   = 50            # sensitivity — higher = less noise
# OVERLAP       = 0             # frame overlap between clips (0 = no overlap)

# USE_YOLO      = False         # True = YOLO segmentation, False = MOG2
# YOLO_MODEL    = "yolov8m-seg.pt"  # medium model — best balance of accuracy/speed
# YOLO_CONF     = 0.15          # lower conf helps detect small/obscured children
# YOLO_IMGSZ    = 1024          # higher resolution = better detection quality
# YOLO_IOU      = 0.45          # handles overlapping people

# SAVE_PREVIEW  = True          # save first clip of each class as .mp4 for visual check
# # ───────────────────────────────────────────────────────────────────────────────


# # ── YOLO Segmentation ─────────────────────────────────────────────────────────

# def load_yolo():
#     """Load YOLO model once and reuse across all videos."""
#     print(f"[YOLO] Loading {YOLO_MODEL}...")
#     model = YOLO(YOLO_MODEL)
#     # Move to GPU if available
#     import torch
#     if torch.cuda.is_available():
#         model.to("cuda")
#         print(f"[YOLO] Running on GPU: {torch.cuda.get_device_name(0)}")
#     else:
#         print("[YOLO] Running on CPU (consider enabling CUDA for speed)")
#     return model


# def yolo_segment_frame(frame_bgr: np.ndarray, model) -> np.ndarray:
#     """
#     Segment a single frame using YOLO instance segmentation.
#     Returns frame with black background, only person(s) visible.

#     Args:
#         frame_bgr : raw OpenCV frame (BGR, uint8)
#         model     : loaded YOLO model

#     Returns:
#         segmented : (H, W, 3) RGB frame, black background
#     """
#     h, w = frame_bgr.shape[:2]
#     black_frame = np.zeros_like(frame_bgr)

#     # ── Fix: convert BGR → RGB before YOLO inference ─────────────────────────
#     frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

#     results = model(
#         frame_rgb,
#         classes=[0],          # person class only
#         conf=YOLO_CONF,
#         imgsz=YOLO_IMGSZ,
#         iou=YOLO_IOU,
#         verbose=False
#     )

#     if results[0].masks is not None:
#         dilate_kernel  = np.ones((20, 20), np.uint8)
#         combined_mask  = np.zeros((h, w), dtype=np.uint8)

#         for mask in results[0].masks.data:
#             mask_np      = mask.cpu().numpy()
#             mask_resized = cv2.resize(mask_np, (w, h))
#             mask_uint8   = (mask_resized > 0.5).astype(np.uint8) * 255

#             # Expand mask to avoid cutting off limbs
#             mask_dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)

#             # Smooth edges
#             mask_blurred = cv2.GaussianBlur(mask_dilated, (21, 21), 0)

#             combined_mask = cv2.bitwise_or(combined_mask, mask_blurred)

#         # Apply mask — keep original BGR colours, black background
#         mask_bool = combined_mask > 127
#         black_frame[mask_bool] = frame_bgr[mask_bool]

#     # Resize to standard 3D CNN input size
#     resized = cv2.resize(black_frame, FRAME_SIZE)
#     return resized


# # ── MOG2 Segmentation (fallback) ──────────────────────────────────────────────

# def build_bg_subtractor():
#     # Use KNN for potentially smoother masks
#     return cv2.createBackgroundSubtractorKNN(
#         history=MOG2_HISTORY,
#         dist2Threshold=MOG2_THRESH * 10,  # adjust threshold
#         detectShadows=False
#     )


# def clean_mask(mask: np.ndarray) -> np.ndarray:
#     """Fill holes and remove speckles from MOG2 mask."""
#     kernel_close = np.ones((9, 9), np.uint8)
#     kernel_open  = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel_open)
#     kernel_dilate = np.ones((3, 3), np.uint8)
#     mask = cv2.dilate(mask, kernel_dilate, iterations=1)
#     # Add stronger Gaussian blur to smooth edges
#     mask = cv2.GaussianBlur(mask, (11, 11), 0)
#     return mask


# def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
#     """Keep only the largest foreground blob (the child)."""
#     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not contours:
#         return mask
#     clean = np.zeros_like(mask)
#     largest = max(contours, key=cv2.contourArea)
#     cv2.drawContours(clean, [largest], -1, 255, thickness=cv2.FILLED)
#     return clean


# def mog2_segment_frame(frame_bgr: np.ndarray, bg_sub) -> np.ndarray:
#     """Segment frame using MOG2 background subtraction."""
#     mask      = bg_sub.apply(frame_bgr)
#     mask      = clean_mask(mask)
#     mask      = keep_largest_contour(mask)
#     segmented = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
#     resized   = cv2.resize(segmented, FRAME_SIZE)
#     return resized


# # ── Clip Extraction ───────────────────────────────────────────────────────────

# def extract_clips(frames: list) -> list:
#     """
#     Split segmented frames into fixed-length clips.
#     Short final clips are padded by repeating the last frame.
#     """
#     clips = []
#     step  = CLIP_LEN - OVERLAP

#     for start in range(0, len(frames), step):
#         clip = frames[start : start + CLIP_LEN]
#         if len(clip) < CLIP_LEN:
#             clip += [clip[-1]] * (CLIP_LEN - len(clip))
#         if len(clip) == CLIP_LEN:
#             clips.append(np.stack(clip, axis=0))  # (T, H, W, 3)

#     return clips


# # ── Preview Saving ────────────────────────────────────────────────────────────

# def save_clip_as_mp4(clip: np.ndarray, output_path: str, fps: int = 8):
#     """Save a (T, H, W, 3) BGR clip as .mp4 for visual inspection."""
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     h, w   = clip.shape[1], clip.shape[2]
#     # Double resolution for smoother preview
#     preview_h, preview_w = h * 2, w * 2
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (preview_w, preview_h))
#     for i in range(clip.shape[0]):
#         frame = cv2.resize(clip[i], (preview_w, preview_h), interpolation=cv2.INTER_CUBIC)
#         writer.write(frame)  # already BGR
#     writer.release()
#     print(f"  [Preview] Saved {output_path}")


# # ── Per-Video Processing ──────────────────────────────────────────────────────

# def process_video(video_path: str, yolo_model=None) -> list:
#     """
#     Full segmentation pipeline for one video.

#     - YOLO mode  : segments each frame with instance masks
#     - MOG2 mode  : warms up background model, then subtracts background

#     Returns list of np.ndarray clips, each shape (T, H, W, 3).
#     """
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print(f"  [WARN] Cannot open {video_path}, skipping.")
#         return []

#     bg_sub = None
#     if not USE_YOLO or yolo_model is None:
#         bg_sub = build_bg_subtractor()
#         # Warmup MOG2
#         for _ in range(WARMUP_FRAMES):
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             bg_sub.apply(frame)
#         cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#     segmented_frames = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if USE_YOLO and yolo_model is not None:
#             seg = yolo_segment_frame(frame, yolo_model)
#         else:
#             seg = mog2_segment_frame(frame, bg_sub)

#         segmented_frames.append(seg)

#     cap.release()

#     if len(segmented_frames) < CLIP_LEN:
#         print(f"  [WARN] Too short ({len(segmented_frames)} frames): {video_path}")
#         return []

#     return extract_clips(segmented_frames)


# # ── Dataset Processing ────────────────────────────────────────────────────────

# def process_dataset():
#     """Walk RAW_DIR, process every video, save clips to OUTPUT_DIR."""
#     raw_root = Path(RAW_DIR)
#     out_root = Path(OUTPUT_DIR)
#     out_root.mkdir(parents=True, exist_ok=True)

#     video_exts = {".mp4", ".avi", ".mov", ".mkv"}
#     class_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])

#     if not class_dirs:
#         print(f"[ERROR] No class subdirectories found in {RAW_DIR}")
#         return

#     # Load YOLO once — reused across all videos
#     yolo_model = load_yolo() if USE_YOLO else None

#     total_clips = 0

#     for class_dir in class_dirs:
#         class_name    = class_dir.name
#         out_class_dir = out_root / class_name
#         out_class_dir.mkdir(parents=True, exist_ok=True)

#         videos = sorted([f for f in class_dir.iterdir() if f.suffix.lower() in video_exts])
#         print(f"\n[CLASS] {class_name} — {len(videos)} videos")

#         save_preview = SAVE_PREVIEW  # save preview for first video only

#         for video_path in tqdm(videos, desc=f"  {class_name}"):
#             clips = process_video(str(video_path), yolo_model=yolo_model)

#             for i, clip in enumerate(clips):
#                 out_path = out_class_dir / f"{video_path.stem}_clip{i:03d}.npy"
#                 np.save(str(out_path), clip)
#                 total_clips += 1

#             # Save first clip of first video as .mp4 for visual check
#             if save_preview and clips:
#                 preview_path = out_root / f"_preview_{class_name}.mp4"
#                 save_clip_as_mp4(clips[0], str(preview_path))
#                 save_preview = False

#     print(f"\n✅ Done. Total clips saved: {total_clips}")
#     print(f"   Output : {OUTPUT_DIR}")
#     if SAVE_PREVIEW:
#         print(f"   Previews saved in {OUTPUT_DIR}/_preview_<class>.mp4 — check these first!")


# if __name__ == "__main__":
#     process_dataset()

"""
preprocessing/segment_videos.py

Segments videos using YOLO instance segmentation (matching Colab script exactly)
and saves cleaned clips as .npy arrays ready for 3D CNN training.

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
            video1_clip000.npy   # shape: (T, H, W, 3)
            video1_clip001.npy
        class_B/
            ...
"""

import cv2
import numpy as np
import os
import imageio
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

# ─── CONFIG ────────────────────────────────────────────────────────────────────
RAW_DIR       = r"C:\Users\22K-4228\Downloads\dataset"
OUTPUT_DIR    = "data/processed"
CLIP_LEN      = 16            # number of frames per clip
FRAME_SIZE    = (112, 112)    # resize each frame to (W, H) for 3D CNN
OVERLAP       = 0             # frame overlap between clips (0 = no overlap)
SAVE_PREVIEW  = False          # save first clip of each class as .mp4 for visual check
# ───────────────────────────────────────────────────────────────────────────────


def load_yolo():
    """Load YOLOv8m-seg model once and reuse across all videos."""
    print("[YOLO] Loading yolov8m-seg.pt...")
    model = YOLO("yolov8m-seg.pt")
    import torch
    if torch.cuda.is_available():
        model.to("cuda")
        print(f"[YOLO] Running on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[YOLO] Running on CPU — this will be slow, consider enabling CUDA")
    return model


def segment_frame(frame: np.ndarray, model, dilate_kernel: np.ndarray) -> np.ndarray:
    """
    Segment a single frame using YOLO — exactly matching the Colab script.

    - Black background
    - yolov8m-seg, conf=0.15, imgsz=1024, iou=0.45
    - Dilate + GaussianBlur mask to avoid cutting off limbs
    - NO BGR->RGB conversion (matches Colab behaviour exactly)

    Returns:
        black_frame : (H, W, 3) frame with only person(s) on black background
    """
    h, w = frame.shape[:2]
    black_frame = np.zeros_like(frame)

    results = model(
        frame,              # raw BGR frame — same as Colab
        classes=[0],        # person only
        conf=0.15,
        imgsz=1024,
        iou=0.45,
        verbose=False
    )

    if results[0].masks is not None:
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        for mask in results[0].masks.data:
            mask_np      = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            mask_uint8   = (mask_resized > 0.5).astype(np.uint8) * 255

            # Expand mask slightly to avoid cutting off limbs
            mask_dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)

            # Smooth edges
            mask_blurred = cv2.GaussianBlur(mask_dilated, (21, 21), 0)

            combined_mask = cv2.bitwise_or(combined_mask, mask_blurred)

        # Apply mask to original frame
        mask_bool = combined_mask > 127
        black_frame[mask_bool] = frame[mask_bool]

    return black_frame


def extract_clips(frames: list) -> list:
    """
    Split segmented frames into fixed-length clips.
    Short final clips are padded by repeating the last frame.
    """
    clips = []
    step  = CLIP_LEN - OVERLAP

    for start in range(0, len(frames), step):
        clip = frames[start : start + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip += [clip[-1]] * (CLIP_LEN - len(clip))
        if len(clip) == CLIP_LEN:
            clips.append(np.stack(clip, axis=0))  # (T, H, W, 3)

    return clips


# def save_preview_mp4(clip: np.ndarray, output_path: str, fps: int = 8):
#     """Save a (T, H, W, 3) clip as .mp4 for visual inspection."""
#     fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#     h, w   = clip.shape[1], clip.shape[2]
#     writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
#     for i in range(clip.shape[0]):
#         writer.write(clip[i])
#     writer.release()
#     print(f"  [Preview] Saved -> {output_path}")

def save_preview_mp4(clip: np.ndarray, output_path: str, fps: int = 8):
    """Save a (T, H, W, 3) clip as .mp4 for visual inspection using imageio."""
    # Initialize imageio writer with high quality
    writer = imageio.get_writer(output_path, fps=fps, quality=9, macro_block_size=None)
    
    # Scale factor for the preview (112 * 5 = 560px)
    preview_size = (clip.shape[2] * 5, clip.shape[1] * 5) 
    
    for i in range(clip.shape[0]):
        # Extract the tiny 112x112 frame
        frame = clip[i]
        
        # Upscale the frame to 560x560 just for the video preview
        # cv2.INTER_NEAREST keeps the pixels sharp instead of muddying them
        frame_large = cv2.resize(frame, preview_size, interpolation=cv2.INTER_NEAREST)
        
        # Convert BGR (OpenCV format) to RGB (Standard format) before saving
        frame_rgb = cv2.cvtColor(frame_large, cv2.COLOR_BGR2RGB)
        
        writer.append_data(frame_rgb)
        
    writer.close()
    print(f"  [Preview] Saved -> {output_path} (Upscaled for viewing)")

def process_video(video_path: str, model) -> list:
    """
    Full segmentation pipeline for one video.

    Steps:
      1. Read frames
      2. Segment each frame with YOLO (matching Colab script exactly)
      3. Resize segmented frame to 112x112
      4. Split into 16-frame clips

    Returns list of np.ndarray clips, each shape (T, H, W, 3).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path}, skipping.")
        return []

    # Dilate kernel defined once per video (same as Colab)
    dilate_kernel     = np.ones((20, 20), np.uint8)
    segmented_frames  = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Segment frame (black background, person only)
        seg = segment_frame(frame, model, dilate_kernel)

        # Resize to 112x112 for 3D CNN
        seg_resized = cv2.resize(seg, FRAME_SIZE)
        segmented_frames.append(seg_resized)

    cap.release()

    if len(segmented_frames) < CLIP_LEN:
        print(f"  [WARN] Too short ({len(segmented_frames)} frames): {video_path}")
        return []

    return extract_clips(segmented_frames)


def process_dataset():
    """Walk RAW_DIR, process every video, save clips to OUTPUT_DIR."""
    raw_root = Path(RAW_DIR)
    out_root = Path(OUTPUT_DIR)
    out_root.mkdir(parents=True, exist_ok=True)

    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    class_dirs = sorted([d for d in raw_root.iterdir() if d.is_dir()])

    if not class_dirs:
        print(f"[ERROR] No class subdirectories found in {RAW_DIR}")
        return

    # Load YOLO once — reused across ALL videos and classes
    model = load_yolo()

    total_clips = 0

    for class_dir in class_dirs:
        class_name    = class_dir.name
        out_class_dir = out_root / class_name
        out_class_dir.mkdir(parents=True, exist_ok=True)

        videos = sorted([
            f for f in class_dir.iterdir()
            if f.suffix.lower() in video_exts
        ])
        print(f"\n[CLASS] {class_name} — {len(videos)} videos")

        save_preview = SAVE_PREVIEW

        for video_path in tqdm(videos, desc=f"  {class_name}"):
            clips = process_video(str(video_path), model)

            for i, clip in enumerate(clips):
                out_path = out_class_dir / f"{video_path.stem}_clip{i:03d}.npy"
                np.save(str(out_path), clip)
                total_clips += 1

            # Save first clip of first video per class as .mp4 preview
            if save_preview and clips:
                preview_path = out_root / f"_preview_{class_name}.mp4"
                save_preview_mp4(clips[0], str(preview_path))
                save_preview = False

    print(f"\nDone. Total clips saved: {total_clips}")
    print(f"   Output  : {OUTPUT_DIR}")
    if SAVE_PREVIEW:
        print(f"   Previews: {OUTPUT_DIR}/_preview_<class>.mp4 — check these before training!")


if __name__ == "__main__":
    process_dataset()