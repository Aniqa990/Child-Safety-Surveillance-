"""
training/predict.py

Run inference on a single video file using the trained model.

Usage:
    python training/predict.py --video path/to/video.mp4
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.segment_videos import process_video
from models.cnn3d import build_model

CHECKPOINT = "checkpoints/best_model.pth"
CLIP_LEN   = 16
MEAN = [0.45, 0.406, 0.225]
STD  = [0.225, 0.224, 0.229]


def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 → (1, 3, T, H, W) float tensor."""
    clip = clip.astype(np.float32) / 255.0
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # (3, T, H, W)
    for c, (mean, std) in enumerate(zip(MEAN, STD)):
        clip[c] = (clip[c] - mean) / std
    return clip.unsqueeze(0)  # (1, 3, T, H, W)


@torch.no_grad()
def predict(video_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(CHECKPOINT, map_location=device)
    classes = ckpt["classes"]
    model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    clips = process_video(video_path)
    if not clips:
        print("[Predict] Could not extract clips from video.")
        return

    # Average predictions across all clips from the video
    all_probs = []
    for clip in clips:
        tensor = clip_to_tensor(clip).to(device)
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx  = int(np.argmax(avg_probs))

    print(f"\n[Predict] Video: {video_path}")
    print(f"[Predict] Prediction: {classes[pred_idx]} ({avg_probs[pred_idx]*100:.1f}% confidence)")
    print(f"\n[Predict] All class probabilities:")
    for cls, prob in sorted(zip(classes, avg_probs), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:<20} {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    args = parser.parse_args()
    predict(args.video)
