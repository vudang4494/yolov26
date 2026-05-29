# YOLOv26 Face Detection - Pure PyTorch Implementation
# Built with 5 Innovations: DFL Removal, NMS-Free O2O, ProgLoss, STAL, MuSGD

## Architecture

YOLOv26 uses a CSPDarknet backbone with RefinedC3k2 blocks and a PAN-FPN neck,
feeding into a dual-head architecture:

- **O2M Branch (One-to-Many)**: Standard detection head using NMS, each GT box can produce multiple predictions
- **O2O Branch (One-to-One)**: NMS-free detection head using Top-K matching, each GT produces exactly one prediction

## Quick Start

### Test the model (no weights needed):

```bash
PYTHON=/Users/vudang/miniconda3/envs/ViT/bin/python
cd /Users/vudang/PythonLab/Yolo26_Face

# Test all scales
$PYTHON scripts/detect.py --model n --device mps

# Benchmark speed
$PYTHON scripts/benchmark.py --scale n --device mps --runs 50 --warmup 10

# Benchmark all scales
$PYTHON scripts/benchmark.py --scale all --device mps --runs 50 --warmup 10
```

### Train the model:

```bash
# Synthetic data (quick test)
$PYTHON scripts/train.py --scale n --epochs 10 --batch 8

# Real COCO-style dataset
$PYTHON scripts/train.py --scale n --epochs 100 --batch 16 --data data.yaml
```

## Project Structure

```
Yolo26_Face/
├── yolo26/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── blocks.py        # Conv, Bottleneck, C3k2, SPPF
│   │   ├── backbone.py      # CSPDarknet backbone
│   │   ├── neck.py          # PAN-FPN neck
│   │   ├── head.py          # Dual head (O2M + O2O)
│   │   ├── loss.py          # YOLOv26Loss with ProgLoss + STAL
│   │   ├── model.py         # Full YOLOv26Model
│   │   └── postprocess.py   # NMS and Top-K post-processing
│   └── optimizers/
│       ├── __init__.py
│       └── musgd.py         # MuSGD optimizer (Newton-Schulz)
├── scripts/
│   ├── detect.py            # Inference script
│   ├── benchmark.py         # Speed benchmark
│   ├── train.py             # Training script (COCO-style YAML + synthetic)
│   └── download_coco.py     # COCO dataset download
├── runs/                    # Checkpoints saved here
└── requirements.txt
```

## Model Scales

| Scale | Params | Architecture | Use Case |
|-------|--------|---------------|----------|
| N | 4.68M | depth=(1,1,3,3,1), ch=(16,32,64,128,256) | Real-time / Edge |
| S | 30.86M | depth=(1,3,6,6,3), ch=(32,64,128,256,512) | Balanced |
| M | 132.52M | depth=(2,4,8,8,4), ch=(60,120,240,480,960) | Accuracy |
| L | 478.53M | depth=(3,6,12,12,6), ch=(96,192,384,768,1536) | High accuracy |

## 5 Innovations

1. **DFL Removal (reg_max=4)**: Replaces 64-channel softmax DFL with 4-channel L1, reducing head params ~4x
2. **NMS-Free O2O**: Top-K matching eliminates NMS post-processing bottleneck
3. **ProgLoss**: Progressive class re-weighting boosts rare classes during training
4. **STAL**: Small-target-aware label assignment for better small object detection
5. **MuSGD**: Newton-Schulz orthogonalization applied to Conv2d weights each step

## Benchmark (Apple M4 MPS)

| Scale | FPS | P99 Latency | Params |
|-------|-----|-------------|--------|
| N | 68.6 | 15.85ms | 4.68M |
| S | 23.6 | 45.28ms | 30.86M |
| M | 7.1 | 146.06ms | 132.52M |
| L | 2.2 | 480.37ms | 478.53M |
