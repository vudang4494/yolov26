#!/usr/bin/env python3
"""
YOLOv26 Benchmark Script - Speed benchmark all scales.
"""
import argparse
import json
import os
import sys
import time
import math
import statistics
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model, SCALE_CONFIGS


def benchmark_single(model, device, img_size, warmup, runs):
    """Benchmark a single model."""
    x = torch.randn(1, 3, img_size, img_size, device=device)
    model.eval()

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)

        latencies = []
        for _ in range(runs):
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            _ = model(x)
            if device == "mps":
                torch.mps.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies = sorted(latencies)
    mean_ms = statistics.mean(latencies)
    median_ms = statistics.median(latencies)
    p50 = latencies[int(len(latencies) * 0.50)]
    p90 = latencies[int(len(latencies) * 0.90)]
    p95 = latencies[int(len(latencies) * 0.95)]
    p99 = latencies[int(len(latencies) * 0.99)]
    fps = 1000.0 / mean_ms

    total_params = sum(p.numel() for p in model.parameters())
    gflops = estimate_gflops(model, img_size)

    return {
        "mean_ms": mean_ms,
        "median_ms": median_ms,
        "p50_ms": p50,
        "p90_ms": p90,
        "p95_ms": p95,
        "p99_ms": p99,
        "fps": fps,
        "params_M": total_params / 1e6,
        "gflops": gflops,
    }


def estimate_gflops(model, img_size):
    """Estimate FLOPs from model parameters and input size."""
    total = 0
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            h_out = img_size // (2 ** (int(name.split(".")[1]) if ".".join(name.split(".")[:2]).split(".")[-1].isdigit() else 0))
            h_out = img_size
            kh, kw = module.kernel_size
            cin, cout = module.in_channels, module.out_channels
            groups = module.groups
            out_elements = h_out * h_out * cout
            flops = 2 * kh * kw * (cin // groups) * out_elements
            total += flops
        elif isinstance(module, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            total += 2 * module.num_features * (img_size * img_size)
    return total / 1e9


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Speed Benchmark")
    parser.add_argument("--scale", type=str, default="n",
                       help="Scale: n, s, m, l, x, or 'all'")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cpu, cuda")
    parser.add_argument("--runs", type=int, default=100,
                       help="Number of benchmark runs")
    parser.add_argument("--warmup", type=int, default=20,
                       help="Warmup runs")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--output", type=str, default="",
                       help="Output JSON file")
    parser.add_argument("--compare-yolo11", action="store_true",
                       help="Compare with YOLO11 from ultralytics")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"

    print(f"{'='*60}")
    print(f"  YOLOv26 Speed Benchmark")
    print(f"  Device: {device}")
    print(f"  Image size: {args.img_size}")
    print(f"  Runs: {args.runs} (warmup: {args.warmup})")
    print(f"{'='*60}")

    results_all = {}

    if args.scale == "all":
        scales = ["n", "s", "m", "l", "x"]
    else:
        scales = [args.scale]

    for scale in scales:
        print(f"\nBenchmarking YOLOv26-{scale.upper()}...")
        model = YOLOv26Model(num_classes=80, scale=scale,
                             image_size=args.img_size).to(device)
        model.eval()

        cfg = SCALE_CONFIGS[scale]
        total_params = sum(p.numel() for p in model.parameters())

        bench = benchmark_single(model, device, args.img_size, args.warmup, args.runs)
        results_all[f"yolo26_{scale}"] = {
            "model": f"YOLO26-{scale.upper()}",
            "scale": scale,
            "params": total_params,
            "params_M": bench["params_M"],
            "gflops": bench["gflops"],
            "mean_ms": bench["mean_ms"],
            "median_ms": bench["median_ms"],
            "p50_ms": bench["p50_ms"],
            "p90_ms": bench["p90_ms"],
            "p95_ms": bench["p95_ms"],
            "p99_ms": bench["p99_ms"],
            "fps": bench["fps"],
            "width_mult": cfg["width_mult"],
        }

        print(f"  Params: {bench['params_M']:.2f}M")
        print(f"  GFLOPs: {bench['gflops']:.1f}G")
        print(f"  Latency: {bench['mean_ms']:.2f}ms mean, {bench['median_ms']:.2f}ms median")
        print(f"  P90: {bench['p90_ms']:.2f}ms | P95: {bench['p95_ms']:.2f}ms | P99: {bench['p99_ms']:.2f}ms")
        print(f"  FPS: {bench['fps']:.1f}")

        del model
        torch.mps.empty_cache() if device == "mps" else None

    if args.compare_yolo11:
        try:
            from ultralytics import YOLO
            print(f"\nBenchmarking YOLO11-N (baseline)...")
            yolo11 = YOLO("yolo11n.pt")
            yolo11.to(device)

            x = torch.randn(1, 3, args.img_size, args.img_size)
            if device == "mps":
                x = x.to("mps")

            yolo11.model.eval()
            with torch.no_grad():
                for _ in range(args.warmup):
                    _ = yolo11.model(x)

                latencies = []
                for _ in range(args.runs):
                    if device == "mps":
                        torch.mps.synchronize()
                    t0 = time.perf_counter()
                    _ = yolo11.model(x)
                    if device == "mps":
                        torch.mps.synchronize()
                    latencies.append((time.perf_counter() - t0) * 1000)

            latencies = sorted(latencies)
            mean_ms = statistics.mean(latencies)
            fps = 1000.0 / mean_ms
            total_params = sum(p.numel() for p in yolo11.model.parameters())

            results_all["yolo11_n"] = {
                "model": "YOLO11-N",
                "params": total_params,
                "params_M": total_params / 1e6,
                "mean_ms": mean_ms,
                "median_ms": statistics.median(latencies),
                "p90_ms": latencies[int(len(latencies) * 0.90)],
                "fps": fps,
            }
            print(f"  YOLO11-N: {mean_ms:.2f}ms ({fps:.1f} FPS)")

        except ImportError:
            print("ultralytics not installed, skipping YOLO11 baseline")

    print(f"\n{'='*60}")
    print(f"{'Model':<12} {'Params':<8} {'GFLOPs':<8} {'Mean(ms)':<10} {'P90(ms)':<10} {'FPS':<8}")
    print(f"{'-'*60}")
    for name, r in sorted(results_all.items(), key=lambda x: x[1]["mean_ms"]):
        gflops_str = f"{r.get('gflops', 0):.1f}G" if "gflops" in r else "—"
        print(f"{r['model']:<12} {r['params_M']:.2f}M    {gflops_str:<8} {r['mean_ms']:<10.2f} {r.get('p90_ms', 0):<10.2f} {r['fps']:<8.1f}")
    print(f"{'='*60}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results_all, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
