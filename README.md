# FYP — Child Action Recognition with 3D CNN

Classifies children's actions from video using:
- **R(2+1)D-18** pretrained 3D CNN (classification)

---

## Project Structure

```
fyp-action-recognition/
├── preprocessing/
│   ├── preprocess_videos.py           # Step 1: .npy files
│   └── split_dataset.py            # Step 2: train/val/test split
├── models/
│   └── cnn3d.py                    # R(2+1)D model definition
├── utils/
│   └── dataset.py                  # PyTorch Dataset + DataLoaders
├── training/
│   ├── train.py                    
│   ├── evaluate.py                 
│   └── train_kfold.py
├── docker/
│   └── Dockerfile
├── requirements.txt
└── README.md
```

---

## Setup

### Local (with NVIDIA GPU)

```bash
# 1. Clone / download this project
cd fyp-action-recognition

# 2. Install dependencies
pip install -r requirements.txt

# 3. Verify GPU is available
python -c "import torch; print(torch.cuda.is_available())"
```


---

## Step-by-Step Usage

### Step 1 — Organise your videos

```
data/raw/
    jumping/
        child1_jump.mp4
        child2_jump.mp4
    waving/
        child1_wave.mp4
    running/
        ...
```

### Step 2 — Preprocess videos (MOG2)

```bash
python preprocessing/preprocess_videos.py
```

This saves `.npy` clip arrays to `data/processed/`.  
Each clip is shape `(16, 112, 112, 3)`.


### Step 3 — Split into train/val/test

```bash
python preprocessing/split_dataset.py
```

Default split: 70% train / 15% val / 15% test.

### Step 4 — Train

```bash
python training/train.py
```

Training uses two phases:
- **Phase 1** (5 epochs): only the classification head is trained
- **Phase 2** (15 epochs): backbone unfrozen from `layer4` for full fine-tuning

Best model is saved to `checkpoints/best_model.pth`.  
Monitor training with TensorBoard:

```bash
tensorboard --logdir runs
```

### Step 5 — Evaluate on test set

```bash
python training/evaluate.py
```

Prints per-class accuracy + saves `results/confusion_matrix.png`.


---

## Pretrained Model

R(2+1)D-18 weights are automatically downloaded from torchvision on first run (~120 MB).  
Pretrained on **Kinetics-400** (400 human action classes) — transfer learning works well for child actions.


