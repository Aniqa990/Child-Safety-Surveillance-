import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ─── KAGGLE CONFIG ────────────────────────────────────────────────────────────
BASE       = Path("/kaggle/input/aniqaazhar990")
OUTPUT_DIR = Path("/kaggle/working/data/processed")
CLIP_LEN   = 16
FRAME_SIZE = (112, 112)

# (dataset_folder_under_BASE, inner_subfolder_name, destination_class)
SOURCES = [
    ("fall-child",   "fall",             "fall"),
    ("fight-child",  "fight",            "fight"),
    ("fight-child",  "additional_fight", "fight"),
    ("adultfight",   "adultFight",       "fight"),
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
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    total_clips = 0

    for root_folder, sub_folder, class_name in SOURCES:
        src_path = BASE / root_folder / sub_folder
        dest_path = OUTPUT_DIR / class_name
        dest_path.mkdir(parents=True, exist_ok=True)
        
        if not src_path.exists():
            print(f"[SKIP] Path not found: {src_path}")
            continue

        videos = [f for f in src_path.iterdir() if f.suffix.lower() in video_exts]
        print(f"\n[PROCESSING] {class_name} from {sub_folder} ({len(videos)} videos)")

        for v_path in tqdm(videos):
            clips = process_video(v_path)
            for i, clip in enumerate(clips):
                out_name = f"{v_path.stem}_clip{i:03d}.npy"
                np.save(str(dest_path / out_name), clip)
                total_clips += 1

    print(f"\nDone. Total clips: {total_clips} saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()