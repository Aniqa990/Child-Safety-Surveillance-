# # # """
# # # training/predict.py

# # # Run inference on a single video file using the trained model.

# # # Usage:
# # #     python training/predict.py --video path/to/video.mp4
# # # """

# # # import os
# # # import sys
# # # import argparse
# # # import torch
# # # import numpy as np
# # # import cv2 # Make sure to import cv2 at the top

# # # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # # from preprocessing.segment_videos import process_video
# # # from models.cnn3d import build_model

# # # CHECKPOINT = "G:/My Drive/Model_Checkpoints/best_r2plus1d.pth"
# # # CLIP_LEN   = 16
# # # MEAN = [0.45, 0.406, 0.225]
# # # STD  = [0.225, 0.224, 0.229]


# # # def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
# # #     """(T, H, W, 3) uint8 → (1, 3, T, H, W) float tensor."""
# # #     clip = clip.astype(np.float32) / 255.0
# # #     clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # (3, T, H, W)
# # #     for c, (mean, std) in enumerate(zip(MEAN, STD)):
# # #         clip[c] = (clip[c] - mean) / std
# # #     return clip.unsqueeze(0)  # (1, 3, T, H, W)


# # # # @torch.no_grad()
# # # # def predict(video_path: str):
# # # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # # #     ckpt = torch.load(CHECKPOINT, map_location=device)
# # # #     classes = ckpt["classes"]
# # # #     model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
# # # #     model.load_state_dict(ckpt["model_state"])
# # # #     model.eval()

# # # #     # clips = process_video(video_path)
# # # #     clips = process_video(video_path, model)
# # # #     if not clips:
# # # #         print("[Predict] Could not extract clips from video.")
# # # #         return

# # # #     # Average predictions across all clips from the video
# # # #     all_probs = []
# # # #     for clip in clips:
# # # #         tensor = clip_to_tensor(clip).to(device)
# # # #         logits = model(tensor)
# # # #         probs  = torch.softmax(logits, dim=1).cpu().numpy()[0]
# # # #         all_probs.append(probs)

# # # #     avg_probs = np.mean(all_probs, axis=0)
# # # #     pred_idx  = int(np.argmax(avg_probs))

# # # #     print(f"\n[Predict] Video: {video_path}")
# # # #     print(f"[Predict] Prediction: {classes[pred_idx]} ({avg_probs[pred_idx]*100:.1f}% confidence)")
# # # #     print(f"\n[Predict] All class probabilities:")
# # # #     for cls, prob in sorted(zip(classes, avg_probs), key=lambda x: -x[1]):
# # # #         bar = "█" * int(prob * 30)
# # # #         print(f"  {cls:<20} {prob*100:5.1f}%  {bar}")

# # # @torch.no_grad()
# # # def predict(video_path: str):
# # #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # #     # 1. Load Model
# # #     ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
# # #     classes = ckpt["classes"]
# # #     model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
# # #     model.load_state_dict(ckpt["model_state"])
# # #     model.eval()

# # #     # 2. Extract Frames manually (Replaces process_video)
# # #     cap = cv2.VideoCapture(video_path)
# # #     frames = []
# # #     while cap.isOpened():
# # #         ret, frame = cap.read()
# # #         if not ret: break
# # #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# # #         frame = cv2.resize(frame, (112, 112)) # Match your model's input size
# # #         frames.append(frame)
# # #     cap.release()

# # #     if len(frames) < CLIP_LEN:
# # #         print(f"[Predict] Video too short. Need {CLIP_LEN} frames.")
# # #         return

# # #     # 3. Create Clips
# # #     # This takes the first 16 frames as one clip. 
# # #     # You can loop this to get multiple clips if the video is long.
# # #     clip = np.array(frames[:CLIP_LEN]) # (T, H, W, C)
    
# # #     # 4. Run Inference
# # #     tensor = clip_to_tensor(clip).to(device)
# # #     logits = model(tensor)
# # #     probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

# # #     # 5. Output Results
# # #     pred_idx = int(np.argmax(probs))
# # #     print(f"\n[Predict] Prediction: {classes[pred_idx]} ({probs[pred_idx]*100:.1f}%)")

# # # if __name__ == "__main__":
# # #     parser = argparse.ArgumentParser()
# # #     parser.add_argument("--video", required=True, help="Path to input video")
# # #     args = parser.parse_args()
# # #     predict(args.video)

# # """
# # training/predict.py

# # Run inference on a single video file using the trained model.
# # Scans the entire video in 16-frame segments.

# # Usage:
# #     python training/predict.py --video path/to/video.mp4
# # """

# # import os
# # import sys
# # import argparse
# # import torch
# # import numpy as np
# # import cv2 

# # sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# # from models.cnn3d import build_model

# # CHECKPOINT = "G:/My Drive/Model_Checkpoints/best_r2plus1d.pth"
# # CLIP_LEN   = 16
# # MEAN = [0.45, 0.406, 0.225]
# # STD  = [0.225, 0.224, 0.229]


# # def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
# #     """(T, H, W, 3) uint8 → (1, 3, T, H, W) float tensor."""
# #     clip = clip.astype(np.float32) / 255.0
# #     clip = torch.from_numpy(clip).permute(3, 0, 1, 2)  # (3, T, H, W)
# #     for c, (mean, std) in enumerate(zip(MEAN, STD)):
# #         clip[c] = (clip[c] - mean) / std
# #     return clip.unsqueeze(0)  # (1, 3, T, H, W)


# # @torch.no_grad()
# # def predict(video_path: str):
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# #     # 1. Load Model
# #     ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
# #     classes = ckpt["classes"]
# #     model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
# #     model.load_state_dict(ckpt["model_state"])
# #     model.eval()

# #     # 2. Extract Frames manually
# #     cap = cv2.VideoCapture(video_path)
# #     frames = []
# #     while cap.isOpened():
# #         ret, frame = cap.read()
# #         if not ret: break
# #         frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #         frame = cv2.resize(frame, (112, 112)) # Match your model's input size
# #         frames.append(frame)
# #     cap.release()

# #     if len(frames) < CLIP_LEN:
# #         print(f"[Predict] Video too short. Need at least {CLIP_LEN} frames.")
# #         return

# #     # 3. Create Clips & 4. Run Inference on the ENTIRE video
# #     all_probs = []
    
# #     # We step through the video 16 frames at a time
# #     for i in range(0, len(frames) - CLIP_LEN + 1, CLIP_LEN):
# #         clip = np.array(frames[i : i + CLIP_LEN]) 
        
# #         tensor = clip_to_tensor(clip).to(device)
# #         logits = model(tensor)
# #         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
# #         all_probs.append(probs)

# #     # 5. Output Results
# #     # Average the probabilities across all segments
# #     avg_probs = np.mean(all_probs, axis=0)
# #     pred_idx = int(np.argmax(avg_probs))
    
# #     print(f"\n[Predict] Video: {video_path}")
# #     print(f"[Predict] Analyzed {len(all_probs)} total 16-frame segments.")
# #     print(f"[Predict] Overall Prediction: {classes[pred_idx]} ({avg_probs[pred_idx]*100:.1f}%)")
    
# #     print(f"\n[Predict] Confidence Breakdown:")
# #     for cls, prob in sorted(zip(classes, avg_probs), key=lambda x: -x[1]):
# #         bar = "█" * int(prob * 30)
# #         print(f"  {cls:<15}: {prob*100:5.1f}%  {bar}")


# # if __name__ == "__main__":
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--video", required=True, help="Path to input video")
# #     args = parser.parse_args()
# #     predict(args.video)

# import os
# import sys
# import argparse
# import torch
# import numpy as np
# import cv2 
# from ultralytics import YOLO

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from models.cnn3d import build_model

# # --- CONFIG ---
# CHECKPOINT = "G:/My Drive/Model_Checkpoints/best_r2plus1d.pth"
# YOLO_MODEL = "yolov8m-seg.pt"
# CLIP_LEN   = 16
# FRAME_SIZE = (112, 112)
# MEAN = [0.45, 0.406, 0.225]
# STD  = [0.225, 0.224, 0.229]

# def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
#     clip = clip.astype(np.float32) / 255.0
#     clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
#     for c, (mean, std) in enumerate(zip(MEAN, STD)):
#         clip[c] = (clip[c] - mean) / std
#     return clip.unsqueeze(0)

# def segment_frame(frame: np.ndarray, yolo_model, dilate_kernel: np.ndarray) -> np.ndarray:
#     """Isolates person on black background using YOLO segmentation."""
#     h, w = frame.shape[:2]
#     black_frame = np.zeros_like(frame)

#     results = yolo_model(frame, classes=[0], conf=0.15, imgsz=1024, iou=0.45, verbose=False)

#     if results[0].masks is not None:
#         combined_mask = np.zeros((h, w), dtype=np.uint8)
#         for mask in results[0].masks.data:
#             mask_np = mask.cpu().numpy()
#             mask_resized = cv2.resize(mask_np, (w, h))
#             mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
#             mask_dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)
#             mask_blurred = cv2.GaussianBlur(mask_dilated, (21, 21), 0)
#             combined_mask = cv2.bitwise_or(combined_mask, mask_blurred)
        
#         mask_bool = combined_mask > 127
#         black_frame[mask_bool] = frame[mask_bool]

#     return black_frame

# @torch.no_grad()
# def predict(video_path: str):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"[Predict] Using device: {device}")

#     # 1. Load Actions Model
#     ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
#     classes = ckpt["classes"]
#     action_model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
#     action_model.load_state_dict(ckpt["model_state"])
#     action_model.eval()

#     # 2. Load YOLO Model
#     print("[Predict] Loading YOLO segmentation model...")
#     yolo_model = YOLO(YOLO_MODEL)
#     if torch.cuda.is_available(): yolo_model.to(device)

#     # 3. Process Video with Segmentation
#     cap = cv2.VideoCapture(video_path)
#     dilate_kernel = np.ones((20, 20), np.uint8)
#     segmented_frames = []

#     print("[Predict] Segmenting video frames...")
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret: break
        
#         # Apply the exact same segmentation used in training
#         seg_frame = segment_frame(frame, yolo_model, dilate_kernel)
#         seg_resized = cv2.resize(seg_frame, FRAME_SIZE)
        
#         # Match training: Convert BGR to RGB
#         seg_rgb = cv2.cvtColor(seg_resized, cv2.COLOR_BGR2RGB)
#         segmented_frames.append(seg_rgb)
#     cap.release()

#     if len(segmented_frames) < CLIP_LEN:
#         print("[Predict] Video too short.")
#         return

#     # 4. Inference on Segments
#     all_probs = []
#     for i in range(0, len(segmented_frames) - CLIP_LEN + 1, CLIP_LEN):
#         clip = np.array(segmented_frames[i : i + CLIP_LEN])
#         tensor = clip_to_tensor(clip).to(device)
#         logits = action_model(tensor)
#         probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
#         all_probs.append(probs)

#     # 5. Results
#     avg_probs = np.mean(all_probs, axis=0)
#     pred_idx = int(np.argmax(avg_probs))
    
#     print(f"\n[Predict] Final Prediction: {classes[pred_idx]} ({avg_probs[pred_idx]*100:.1f}%)")
#     for cls, prob in sorted(zip(classes, avg_probs), key=lambda x: -x[1]):
#         print(f"  {cls:<15}: {prob*100:5.1f}% {'█' * int(prob * 30)}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--video", required=True)
#     args = parser.parse_args()
#     predict(args.video)

import os
import sys
import argparse
import torch
import numpy as np
import cv2 
from pathlib import Path
from ultralytics import YOLO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.cnn3d import build_model

# --- CONFIG ---
CHECKPOINT = "G:/My Drive/Model_Checkpoints/best_r2plus1d.pth"
YOLO_MODEL = "yolov8m-seg.pt"
CLIP_LEN   = 16
FRAME_SIZE = (112, 112)
MEAN = [0.45, 0.406, 0.225]
STD  = [0.225, 0.224, 0.229]
VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}

def clip_to_tensor(clip: np.ndarray) -> torch.Tensor:
    clip = clip.astype(np.float32) / 255.0
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)
    for c, (mean, std) in enumerate(zip(MEAN, STD)):
        clip[c] = (clip[c] - mean) / std
    return clip.unsqueeze(0)

def segment_frame(frame: np.ndarray, yolo_model, dilate_kernel: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    black_frame = np.zeros_like(frame)
    results = yolo_model(frame, classes=[0], conf=0.15, imgsz=1024, iou=0.45, verbose=False)

    if results[0].masks is not None:
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for mask in results[0].masks.data:
            mask_np = mask.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (w, h))
            mask_uint8 = (mask_resized > 0.5).astype(np.uint8) * 255
            mask_dilated = cv2.dilate(mask_uint8, dilate_kernel, iterations=1)
            mask_blurred = cv2.GaussianBlur(mask_dilated, (21, 21), 0)
            combined_mask = cv2.bitwise_or(combined_mask, mask_blurred)
        
        mask_bool = combined_mask > 127
        black_frame[mask_bool] = frame[mask_bool]
    return black_frame

@torch.no_grad()
def run_prediction(video_path, action_model, yolo_model, classes, device):
    """Processes a single video and returns the prediction results."""
    cap = cv2.VideoCapture(str(video_path))
    dilate_kernel = np.ones((20, 20), np.uint8)
    segmented_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        seg_frame = segment_frame(frame, yolo_model, dilate_kernel)
        seg_resized = cv2.resize(seg_frame, FRAME_SIZE)
        seg_rgb = cv2.cvtColor(seg_resized, cv2.COLOR_BGR2RGB)
        segmented_frames.append(seg_rgb)
    cap.release()

    if len(segmented_frames) < CLIP_LEN:
        return None, f"Too short ({len(segmented_frames)} frames)"

    all_probs = []
    for i in range(0, len(segmented_frames) - CLIP_LEN + 1, CLIP_LEN):
        clip = np.array(segmented_frames[i : i + CLIP_LEN])
        tensor = clip_to_tensor(clip).to(device)
        logits = action_model(tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs.append(probs)

    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(np.argmax(avg_probs))
    return classes[pred_idx], avg_probs[pred_idx]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to a video file OR a folder of videos")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Init] Using device: {device}")

    # 1. Load Models Once
    ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    action_model = build_model(num_classes=len(classes), freeze_backbone=False).to(device)
    action_model.load_state_dict(ckpt["model_state"])
    action_model.eval()

    print("[Init] Loading YOLOv8m-seg...")
    yolo_model = YOLO(YOLO_MODEL)
    if torch.cuda.is_available(): yolo_model.to(device)

    # 2. Identify Files to Process
    input_path = Path(args.input)
    video_files = []

    if input_path.is_file():
        video_files.append(input_path)
    elif input_path.is_dir():
        video_files = [f for f in input_path.iterdir() if f.suffix.lower() in VIDEO_EXTS]
        print(f"[Init] Found {len(video_files)} videos in folder: {input_path.name}")
    else:
        print("[Error] Input path does not exist.")
        return

    # 3. Loop through all videos
    print("-" * 50)
    for vid in video_files:
        print(f"[Processing] {vid.name}...")
        label, confidence = run_prediction(vid, action_model, yolo_model, classes, device)
        
        if label:
            conf_percent = confidence * 100
            bar = "█" * int(confidence * 20)
            print(f"  Result: {label:<15} ({conf_percent:>5.1f}%) {bar}")
        else:
            print(f"  Result: Skipped ({confidence})")
    print("-" * 50)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()