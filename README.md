# YOLOv26 Face Detection - Pure PyTorch Implementation
# Built with 5 Innovations: DFL Removal, NMS-Free O2O, ProgLoss, STAL, MuSGD

## Architecture

YOLOv26 uses a CSPDarknet backbone with RefinedC3k2 blocks and a PAN-FPN neck,
feeding into a dual-head architecture:

- **O2M Branch (One-to-Many)**: Standard detection head using NMS, each GT box can produce multiple predictions
- **O2O Branch (One-to-One)**: NMS-free detection head using Hungarian matching, each GT produces exactly one prediction

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
$PYTHON scripts/benchmark.py --scale all --device mps --runs 50 --warmup 10 --output results/benchmark.json
```

## Project Structure

```
Yolo26_Face/
├── yolo26/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── blocks.py        # Conv, Bottleneck, C3k2, SPPF, PositionalEmbedding
│   │   ├── backbone.py       # CSPDarknet backbone
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
│   ├── train.py             # Training script
│   ├── evaluate.py          # mAP evaluation
│   └── download_coco.py     # COCO dataset download
└── requirements.txt
```

## Model Scales

| Scale | Params | GFLOPs | Notes |
|-------|--------|--------|-------|
| N | 7.2M | 2.4G | Nano - fastest, mobile |
| S | 28.8M | 9.6G | Small - balanced |
| M | 64.9M | 21.6G | Medium - accuracy |
| L | 115.3M | 38.3G | Large - high accuracy |
| X | 180.1M | 59.9G | Extra-large - max accuracy |
