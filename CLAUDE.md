# CLAUDE.md — YOLOv26 Pure PyTorch Implementation

## Project Overview

**YOLOv26** là implementation thuần PyTorch của mô hình object detection với 5 innovations:
1. **DFL Removal** — reg_max=4 (thay vì 16) giảm 4x channels
2. **NMS-Free O2O** — Hungarian matching thay NMS, inference nhanh hơn
3. **ProgLoss** — progressive class re-weighting cho class imbalance
4. **STAL** — small-target-aware label assignment
5. **MuSGD** — Newton-Schulz optimizer cho matrix parameters

**Repository:** `/Users/vudang/PythonLab/Yolo26_Face`
**Python:** `/Users/vudang/miniconda3/envs/ViT/bin/python`
**Device:** Apple Silicon MPS (M4 GPU)

## Architecture

```
YOLOv26Model
├── Backbone: CSPDarknet + RefinedC3k2 + SPPF
│   └── P3(80x80), P4(40x40), P5(20x20)
├── Neck: PAN-FPN (top-down FPN + bottom-up PAN)
└── Head: Dual Head
    ├── O2M branch: 4xreg_max channels → DFL softmax → NMS
    └── O2O branch: 4 channels → direct sigmoid → Top-K (NMS-free)
```

**Model Scales:** N(5.7M) / S(30.9M) / M(132.5M) / L(478.5M) / X(1.3B)
**Benchmark:** YOLO26-N @ 61 FPS trên Apple M4 MPS

## Commands

### Inference
```bash
# Quick test
python scripts/detect.py --model n --device mps

# Webcam
python scripts/detect.py --model n --device mps --source 0

# Image
python scripts/detect.py --model n --device mps --source /path/to/image.jpg

# Full pipeline
python scripts/detect.py --model s --device mps --conf 0.3 --iou 0.45
```

### Benchmark
```bash
# Single scale
python scripts/benchmark.py --scale n --device mps --runs 100 --warmup 20

# All scales
python scripts/benchmark.py --scale all --device mps --runs 100 --warmup 20

# Compare with YOLO11 baseline
python scripts/benchmark.py --scale n --device mps --compare-yolo11
```

### Training
```bash
# Basic training
python scripts/train.py --scale n --epochs 10 --batch 8 --device mps

# Full training
python scripts/train.py --scale n --epochs 300 --batch 16 --device mps --lr 0.01

# Custom dataset (COCO format)
python scripts/train.py --scale n --epochs 100 --batch 16 --device mps --data dataset/coco
```

### Evaluation
```bash
# mAP evaluation
python scripts/evaluate.py --scale n --device mps --num-images 500

# With trained weights
python scripts/evaluate.py --scale n --device mps --weights runs/yolo26_n_final.pt
```

### Dataset
```bash
# Download COCO val (5000 images)
python scripts/download_coco.py --split val --limit 5000

# Download COCO train (full)
python scripts/download_coco.py --split train --limit 0

# Small test set (8 images)
python scripts/download_coco.py --split val --limit 8
```

## Project Structure

```
Yolo26_Face/
├── yolo26/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── blocks.py        # Conv, Bottleneck, C3k2, SPPF
│   │   ├── backbone.py       # CSPDarknet backbone
│   │   ├── neck.py          # PAN-FPN neck
│   │   ├── head.py          # Dual head (O2M + O2O)
│   │   ├── loss.py          # ProgLoss + STAL + Hungarian
│   │   ├── model.py         # Full YOLOv26Model + anchors
│   │   └── postprocess.py   # NMS + Top-K post-processing
│   └── optimizers/
│       └── musgd.py         # MuSGD optimizer
├── scripts/
│   ├── detect.py            # Inference
│   ├── benchmark.py         # Speed benchmark
│   ├── train.py             # Training
│   ├── evaluate.py          # mAP evaluation
│   └── download_coco.py     # Dataset download
├── requirements.txt
├── README.md
└── CLAUDE.md                # This file
```

## Key Classes

| Class | Location | Purpose |
|-------|----------|---------|
| `YOLOv26Model` | `yolo26/core/model.py` | Full model: backbone + neck + head |
| `C3k2` | `yolo26/core/blocks.py` | CSP bottleneck stack |
| `YOLOv26Head` | `yolo26/core/head.py` | Dual head: O2M + O2O |
| `YOLOv26Loss` | `yolo26/core/loss.py` | Combined O2M + O2O loss |
| `PostProcess` | `yolo26/core/postprocess.py` | O2O decode + NMS |
| `MuSGD` | `yolo26/optimizers/musgd.py` | Custom optimizer |

## Development Rules

1. **Device**: Luôn check MPS available, fallback to CPU
   ```python
   device = 'mps' if torch.backends.mps.is_available() else 'cpu'
   ```
2. **Model output**: Training mode trả `(o2m_preds, o2o_preds)`, inference trả `list[dict]`
3. **Anchors**: Tự động generate trong `__init__`, dùng cho cả training và inference
4. **Loss**: Dùng `YOLOv26Loss` cho training, `PostProcess` cho inference
5. **Dependencies**: Chỉ import từ `torch`, `torchvision`, `scipy`, `cv2` — không Ultralytics
6. **Scale naming**: Luôn dùng lowercase `n/s/m/l/x`
7. **Num classes**: Default 80 (COCO), có thể thay đổi cho custom dataset

## Troubleshooting

| Vấn đề | Giải pháp |
|---------|-----------|
| `MPS OOM` | Giảm `--batch` hoặc dùng scale `n` |
| `ModuleNotFoundError` | Dùng `/Users/vudang/miniconda3/envs/ViT/bin/python` |
| `Dataset download fails` | Kiểm tra HuggingFace token trong `~/.cache/huggingface/` |
| `Benchmark chậm` | Giảm `--runs`, tăng `--warmup` |
| `Loss NaN` | Giảm `--lr` (try 0.001 thay vì 0.01) |

## Fine-tuning for Face Detection

Để fine-tune cho face detection (1 class thay vì 80):

```python
from yolo26.core.model import YOLOv26Model
model = YOLOv26Model(num_classes=1, scale='n', image_size=640)
```

Sau đó train với dataset face:
```bash
python scripts/train.py --scale n --epochs 50 --batch 8 --num-classes 1 --data /path/to/face/dataset
```
