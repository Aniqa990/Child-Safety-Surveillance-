import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── KAGGLE CONFIG ────────────────────────────────────────────────────────────
BASE       = Path("/kaggle/input/datasets/aniqaazhar990")
OUTPUT_DIR = Path("/kaggle/working/data/processed")
CLIP_LEN   = 16
FRAME_SIZE = (112, 112)

# (dataset_folder_under_BASE, inner_subfolder_name, destination_class)
SOURCES = [
    ("fall-child",   "fall",             "fall"),
    ("fight-child",  "fight",            "fight"),
    ("fight-child",  "additional_fight", "fight"),
    ("unsafeclimb",  "unsafeClimb",      "unsafeClimb"),
    ("unsafeclimb",  "additional_climb", "unsafeClimb"),
    ("unsafethrow",  "unsafeThrow",      "unsafeThrow"),
    ("unsafethrow",  "additional_throw", "unsafeThrow"),
    ("unsafejump",   "unsafeJump",       "unsafeJump"),
    ("unsafejump",   "additional_jump",  "unsafeJump"),
]
# ──────────────────────────────────────────────────────────────────────────────

def extract_clips(frames):
    clips = []
    for start in range(0, len(frames), CLIP_LEN):
        clip = frames[start : start + CLIP_LEN]
        if len(clip) < CLIP_LEN:
            clip = clip + [clip[-1]] * (CLIP_LEN - len(clip))
        clips.append(np.stack(clip, axis=0))
    return clips

def process_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame_rgb, FRAME_SIZE))
    cap.release()
    return extract_clips(frames) if len(frames) >= CLIP_LEN else []

def process_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    total_clips = 0

    print(f"Searching for videos in: {BASE}")

    for dataset_folder, inner_name, class_name in SOURCES:
        src_base = BASE / dataset_folder
        dest_path = OUTPUT_DIR / class_name
        dest_path.mkdir(parents=True, exist_ok=True)
        
        if not src_base.exists():
            print(f"[WARN] Root folder not found: {src_base}")
            continue

        # FIND FOLDER LOGIC: handles timestamped wrappers by searching recursively
        found_dirs = [p for p in src_base.rglob(inner_name) if p.is_dir()]
        if not found_dirs:
            print(f"[WARN] No subfolder '{inner_name}' found under {src_base}")
            continue

        for src_dir in found_dirs:
            videos = [f for f in src_dir.iterdir() if f.suffix.lower() in VIDEO_EXTS]
            if not videos:
                continue

            print(f"\n[PROCESSING] {class_name} from {src_dir.relative_to(BASE)} ({len(videos)} videos)")

            for v_path in tqdm(videos, desc=f"  {class_name}"):
                clips = process_video(v_path)
                for i, clip in enumerate(clips):
                    # Output name preserves the original filename (including aug_ prefix)
                    out_name = f"{v_path.stem}_clip{i:03d}.npy"
                    np.save(str(dest_path / out_name), clip)
                    total_clips += 1

    print(f"\nPreprocessing Complete. Total clips: {total_clips} saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
