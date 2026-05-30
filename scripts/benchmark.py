#!/usr/bin/env python3
"""
YOLOv26 Comprehensive Benchmark Suite
====================================
Measures: FPS, latency (mean/p50/p90/p99), memory, throughput.
Supports: all scales (N/S/M/L), multiple devices (MPS/CPU/CUDA).

Usage:
    python scripts/benchmark.py                    # All scales, MPS
    python scripts/benchmark.py --scale n         # Single scale
    python scripts/benchmark.py --device cpu      # CPU fallback
    python scripts/benchmark.py --runs 200        # More runs for accuracy
"""

import argparse
import gc
import sys
import time
import torch
import numpy as np

sys.path.insert(0, ".")
from yolo26.core.model import YOLOv26Model, SCALE_CONFIGS


def get_device(requested: str) -> str:
    if requested == "mps":
        if torch.backends.mps.is_available():
            return "mps"
        print(f"  [WARN] MPS not available, falling back to CPU")
    if requested == "cuda" or requested == "gpu":
        if torch.cuda.is_available():
            return "cuda"
        print(f"  [WARN] CUDA not available, falling back to CPU")
    return "cpu"


def clear_cache(device: str):
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()


def benchmark_scale(
    model,
    device: str,
    batch_size: int = 1,
    image_size: int = 640,
    runs: int = 100,
    warmup: int = 30,
) -> dict:
    """Run comprehensive benchmark for a single model configuration."""
    model.eval()
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)

    clear_cache(device)

        # Warmup — critical for accurate measurement
        with torch.no_grad():
            for _ in range(warmup):
                if hasattr(model, 'predict'):
                    _ = model.predict(x)
                else:
                    _ = model(x)

    if device in ("mps", "cuda"):
        if device == "mps":
            torch.mps.synchronize()
        else:
            torch.cuda.synchronize()

    clear_cache(device)

    # Benchmark loop
    latencies = []
    with torch.no_grad():
        for _ in range(runs):
            t0 = time.perf_counter()
            if hasattr(model, 'predict'):
                _ = model.predict(x)
            else:
                _ = model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                try:
                    torch.mps.synchronize()
                except AttributeError:
                    pass
            latencies.append((time.perf_counter() - t0) * 1000)

    latencies = np.array(latencies)
    latencies_ms = latencies  # already in ms

    return {
        "mean_ms": float(np.mean(latencies_ms)),
        "std_ms": float(np.std(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p90_ms": float(np.percentile(latencies_ms, 90)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "p99_ms": float(np.percentile(latencies_ms, 99)),
        "min_ms": float(np.min(latencies_ms)),
        "max_ms": float(np.max(latencies_ms)),
        "fps": float(1000.0 / np.mean(latencies_ms)),
        "throughput_imgs_per_sec": float(
            batch_size * 1000.0 / np.mean(latencies_ms)
        ),
    }


def measure_memory(model, device: str, batch_size: int = 1, image_size: int = 640) -> dict:
    """Measure peak memory usage."""
    clear_cache(device)

    if device == "cpu":
        import tracemalloc
        tracemalloc.start()
        model.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size)
        with torch.no_grad():
            _ = model(x)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return {
            "allocated_mb": round(current / 1024 / 1024, 1),
            "peak_mb": round(peak / 1024 / 1024, 1),
        }

    if device == "mps":
        torch.mps.empty_cache()
        model.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            _ = model(x)
        # MPS doesn't expose direct memory stats like CUDA
        # Use a proxy: count parameters
        param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
        return {
            "model_params_mb": round(param_mb, 1),
            "note": "MPS doesn't expose GPU memory stats. Value = model params only.",
        }

    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model.to(device)
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        with torch.no_grad():
            _ = model(x)
        mem_allocated = torch.cuda.memory_allocated() / 1024 / 1024
        mem_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        mem_peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        return {
            "allocated_mb": round(mem_allocated, 1),
            "reserved_mb": round(mem_reserved, 1),
            "peak_mb": round(mem_peak, 1),
        }


def measure_model_info(model, scale: str) -> dict:
    """Collect model architecture info."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    cfg = model.cfg

    # FLOPs estimate (Conv: H*W*K*K*Cin*Cout*B)
    # Very rough: params * H * W / 1000 for 640x640 input
    flops_estimate = total * 640 * 640 / 1e9  # in billions

    return {
        "scale": scale,
        "params_m": round(total / 1e6, 2),
        "trainable_m": round(trainable / 1e6, 2),
        "channels": cfg["channels"],
        "depths": cfg["depths"],
        "width_mult": cfg["width_mult"],
        "gflops_estimate": round(flops_estimate, 1),
    }


def format_results_table(results: list) -> str:
    """Format benchmark results as ASCII table."""
    header = (
        f"  {'Scale':<6} "
        f"{'Params':<8} "
        f"{'Mean±Std':<12} "
        f"{'P50':<8} "
        f"{'P90':<8} "
        f"{'P99':<8} "
        f"{'FPS':<8} "
        f"{'Throughput':<12} "
        f"{'Device':<6}"
    )
    sep = "  " + "-" * len(header)
    lines = [header, sep]
    for r in results:
        line = (
            f"  {r['scale'].upper():<6} "
            f"{r['params_m']}M  "
            f"{r['mean_ms']:.2f}±{r['std_ms']:.2f}  "
            f"{r['p50_ms']:<8.2f} "
            f"{r['p90_ms']:<8.2f} "
            f"{r['p99_ms']:<8.2f} "
            f"{r['fps']:<8.1f} "
            f"{r['throughput']:<12.1f} "
            f"{r['device']:<6}"
        )
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Benchmark Suite")
    parser.add_argument("--scale", type=str, default="all",
                        choices=["all", "n", "s", "m", "l"],
                        help="Model scale (default: all)")
    parser.add_argument("--device", type=str, default="mps",
                        choices=["mps", "cuda", "cpu"],
                        help="Device (default: mps)")
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of benchmark runs (default: 100)")
    parser.add_argument("--warmup", type=int, default=30,
                        help="Warmup iterations (default: 30)")
    parser.add_argument("--batch", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Input image size (default: 640)")
    parser.add_argument("--num-classes", type=int, default=80,
                        help="Number of classes (default: 80)")
    parser.add_argument("--compare", action="store_true",
                        help="Compare all scales side by side")
    args = parser.parse_args()

    device = get_device(args.device)
    scales = ["n", "s", "m", "l"] if args.scale == "all" else [args.scale]

    print("=" * 90)
    print("  YOLOv26 BENCHMARK SUITE")
    print("=" * 90)
    print(f"  Device:       {device}")
    print(f"  Runs:         {args.runs}")
    print(f"  Warmup:       {args.warmup}")
    print(f"  Batch size:   {args.batch}")
    print(f"  Image size:   {args.image_size}x{args.image_size}")
    print(f"  Num classes:  {args.num_classes}")
    print(f"  PyTorch:     {torch.__version__}")
    print(f"  Date:         {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 90)

    all_results = []

    for scale in scales:
        print(f"\n  {'─' * 86}")
        print(f"  Benchmarking YOLO26-{scale.upper()} ...")
        print(f"  {'─' * 86}")

        # Build model
        model = YOLOv26Model(
            num_classes=args.num_classes,
            scale=scale,
            image_size=args.image_size,
        )
        model.to(device)

        # Model info
        info = measure_model_info(model, scale)
        print(f"  Channels:     {info['channels']}")
        print(f"  Depths:       {info['depths']}")
        print(f"  Width mult:   {info['width_mult']}")
        print(f"  Params:       {info['params_m']}M")
        print(f"  Trainable:    {info['trainable_m']}M")
        print(f"  FLOPs (est):  {info['gflops_estimate']}G (640x640 input)")

        # Memory
        mem = measure_memory(model, device, args.batch, args.image_size)
        print(f"  Memory:       {mem}")

        # Speed benchmark
        print(f"  Running {args.runs} iterations (warmup={args.warmup})...")
        perf = benchmark_scale(
            model,
            device,
            batch_size=args.batch,
            image_size=args.image_size,
            runs=args.runs,
            warmup=args.warmup,
        )

        print(f"\n  ── Speed Results ──")
        print(f"  Mean latency:  {perf['mean_ms']:.2f} ± {perf['std_ms']:.2f} ms")
        print(f"  P50 latency:   {perf['p50_ms']:.2f} ms")
        print(f"  P90 latency:   {perf['p90_ms']:.2f} ms")
        print(f"  P95 latency:   {perf['p95_ms']:.2f} ms")
        print(f"  P99 latency:   {perf['p99_ms']:.2f} ms")
        print(f"  Min latency:   {perf['min_ms']:.2f} ms")
        print(f"  Max latency:   {perf['max_ms']:.2f} ms")
        print(f"  FPS:           {perf['fps']:.1f}")
        print(f"  Throughput:    {perf['throughput_imgs_per_sec']:.1f} imgs/s")

        # Collect for table
        result = {
            "scale": scale,
            "params_m": info["params_m"],
            "gflops": info["gflops_estimate"],
            "mean_ms": perf["mean_ms"],
            "std_ms": perf["std_ms"],
            "p50_ms": perf["p50_ms"],
            "p90_ms": perf["p90_ms"],
            "p95_ms": perf["p95_ms"],
            "p99_ms": perf["p99_ms"],
            "min_ms": perf["min_ms"],
            "max_ms": perf["max_ms"],
            "fps": perf["fps"],
            "throughput": perf["throughput_imgs_per_sec"],
            "device": device,
            "memory": mem,
        }
        all_results.append(result)

        # Cleanup
        del model
        clear_cache(device)

    # Summary table
    if len(all_results) > 1:
        print(f"\n{'=' * 90}")
        print("  SUMMARY TABLE")
        print("=" * 90)
        print(format_results_table(all_results))

    # FPS vs Params summary
    if len(all_results) > 1:
        print(f"\n  FPS / PARAM COUNT TRADEOFF")
        print(f"  {'─' * 50}")
        for r in all_results:
            bar_len = int(r["fps"] / 2)
            bar = "█" * bar_len + "░" * (50 - bar_len)
            print(f"  YOLO26-{r['scale'].upper():<3}  {r['fps']:5.1f} FPS  {r['params_m']:5.1f}M params  {bar}")

    print(f"\n{'=' * 90}")
    print(f"  Benchmark complete. {sum(r['fps'] for r in all_results):.1f} total FPS across {len(all_results)} scales.")
    print("=" * 90)

    return all_results


if __name__ == "__main__":
    results = main()
