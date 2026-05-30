#!/usr/bin/env python3
"""
YOLOv26 ONNX Export
==================
Exports the YOLOv26 backbone+neck+head to ONNX format for CPU/edge inference.

Usage:
    python scripts/export_onnx.py                    # Export all scales
    python scripts/export_onnx.py --scale n         # Single scale
    python scripts/export_onnx.py --opset 18        # Custom opset
"""

import argparse
import os
import sys

sys.path.insert(0, ".")
import torch
from yolo26.core.model import YOLOv26Model


class ONNXWrapper(torch.nn.Module):
    """
    Wraps the backbone+neck+head into a single ONNX-compatible module.
    Postprocess is handled outside ONNX (in Python or a separate runtime).
    """
    def __init__(self, model):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck
        self.head = model.head

    def forward(self, x):
        p = self.backbone(x)
        n = self.neck(p)
        o2m, o2o = self.head(n)
        # Return flattened raw predictions for each level
        outputs = []
        for branch in (o2m, o2o):
            for level in branch:
                outputs.append(level["cls"])
                outputs.append(level["reg"])
        return outputs


def export_onnx(scale="n", num_classes=80, image_size=640,
                opset=18, output_dir="."):
    """Export YOLO26 scale to ONNX."""
    print(f"Building YOLO26-{scale.upper()}...")
    model = YOLOv26Model(scale=scale, num_classes=num_classes,
                         image_size=image_size)
    model.eval()

    wrapper = ONNXWrapper(model)

    output_names = []
    branch_names = ["o2m", "o2o"]
    level_names = ["p3", "p4", "p5"]
    for b in branch_names:
        for l in level_names:
            output_names.append(f"{b}_{l}_cls")
            output_names.append(f"{b}_{l}_reg")

    filename = f"yolo26{scale}.onnx"
    filepath = os.path.join(output_dir, filename)

    x = torch.randn(1, 3, image_size, image_size)
    print(f"Exporting to {filepath} (opset={opset})...")
    torch.onnx.export(
        wrapper, x, filepath,
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        do_constant_folding=True,
    )

    size_mb = os.path.getsize(filepath) / 1e6
    print(f"  File size: {size_mb:.1f}MB")
    print(f"  Outputs: {output_names}")
    return filepath


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 ONNX Export")
    parser.add_argument("--scale", type=str, default="all",
                        choices=["all", "n", "s", "m", "l"])
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--output-dir", type=str, default=".")
    args = parser.parse_args()

    scales = ["n", "s", "m", "l"] if args.scale == "all" else [args.scale]

    print("=" * 60)
    print("  YOLOv26 ONNX Export")
    print("=" * 60)
    print(f"  Classes:    {args.num_classes}")
    print(f"  Image size: {args.image_size}")
    print(f"  Opset:      {args.opset}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    for scale in scales:
        filepath = export_onnx(
            scale=scale,
            num_classes=args.num_classes,
            image_size=args.image_size,
            opset=args.opset,
            output_dir=args.output_dir,
        )
        print()

    print("Export complete!")


if __name__ == "__main__":
    main()
