#!/usr/bin/env python3
"""
YOLOv26 Detect Script - Run inference on images or video.
"""
import argparse
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model
from yolo26.core.postprocess import PostProcess


def draw_boxes(image, results, class_names=None, conf_thresh=0.25):
    """Draw detection boxes on image."""
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]

    for det in results:
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if boxes.numel() == 0:
            continue

        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy() if scores.is_cuda else scores.numpy()
        labels_np = labels.cpu().numpy() if labels.is_cuda else labels.numpy()

        H, W = image.shape[:2]
        boxes_np[:, [0, 2]] *= W
        boxes_np[:, [1, 3]] *= H

        for box, score, label in zip(boxes_np, scores_np, labels_np):
            x1, y1, x2, y2 = box.astype(int)
            cls_id = int(label)
            color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_names[cls_id]}: {score:.2f}"
            cv2.putText(image, label_text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def load_image(path, size=640):
    """Load and resize image to model input size."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    orig_h, orig_w = img.shape[:2]

    scale = min(size / orig_w, size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    canvas = canvas[:, :, ::-1].transpose(2, 0, 1) / 255.0
    canvas = np.ascontiguousarray(canvas, dtype=np.float32)
    return torch.from_numpy(canvas).unsqueeze(0), img, (new_w, new_h), (orig_w, orig_h), scale


def detect_image(model, image_path, device="mps", conf_thresh=0.25, iou_thresh=0.45):
    """Run detection on a single image."""
    model.eval()
    x, orig_img, (new_w, new_h), (orig_w, orig_h), scale = load_image(image_path)
    x = x.to(device)

    with torch.no_grad():
        start = time.time()
        results = model(x)
        latency = (time.time() - start) * 1000

    H, W = orig_img.shape[:2]
    for det in results:
        det["boxes"][:, [0, 2]] /= scale
        det["boxes"][:, [1, 3]] /= scale
        det["boxes"][:, 0].clamp_(0, W)
        det["boxes"][:, 1].clamp_(0, H)
        det["boxes"][:, 2].clamp_(0, W)
        det["boxes"][:, 3].clamp_(0, H)

    result_img = draw_boxes(orig_img.copy(), results)

    fps = 1000.0 / latency if latency > 0 else 0
    num_dets = sum(r["boxes"].shape[0] for r in results)
    print(f"  Image: {image_path}")
    print(f"  Latency: {latency:.1f}ms ({fps:.1f} FPS)")
    print(f"  Detections: {num_dets}")

    return result_img, results, latency


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Detection")
    parser.add_argument("--source", type=str, default="",
                       help="Image path, directory, or '0' for webcam")
    parser.add_argument("--model", type=str, default="n",
                       help="Model scale: n, s, m, l, x")
    parser.add_argument("--weights", type=str, default="",
                       help="Path to pretrained weights (optional)")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cpu, cuda")
    parser.add_argument("--conf", type=float, default=0.25,
                       help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="IoU threshold for NMS")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--save", type=str, default="",
                       help="Output directory to save results")
    parser.add_argument("--class-names", type=str, default="",
                       help="Path to class names text file")
    args = parser.parse_args()

    device = args.device if torch.backends.mps.is_available() and args.device == "mps" else "cpu"
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"

    print(f"Building YOLOv26-{args.model.upper()} on {device}...")
    model = YOLOv26Model(
        num_classes=80,
        scale=args.model,
        image_size=args.img_size,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
    ).to(device)

    if args.weights:
        print(f"Loading weights from {args.weights}")
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=False)

    model.info()

    class_names = None
    if args.class_names:
        with open(args.class_names) as f:
            class_names = [l.strip() for l in f]

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if args.source == "" or args.source == "0":
        print("No source provided. Running quick model test instead...")
        x = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            t0 = time.time()
            results = model(x)
            t1 = time.time()
        print(f"Forward pass: {(t1-t0)*1000:.1f}ms")
        print(f"Output type: {type(results)}")
        if isinstance(results, list):
            print(f"Detections: {len(results[0])} items")
        return

    if os.path.isdir(args.source):
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(Path(args.source).glob(ext))
        print(f"Found {len(image_paths)} images")
    else:
        image_paths = [Path(args.source)]

    total_time = 0
    for img_path in image_paths:
        try:
            result_img, _, latency = detect_image(
                model, str(img_path), device, args.conf, args.iou
            )
            total_time += latency

            if save_dir:
                out_path = save_dir / f"result_{img_path.name}"
                cv2.imwrite(str(out_path), result_img)
                print(f"  Saved: {out_path}")

            cv2.imshow("YOLOv26 Detection", result_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    cv2.destroyAllWindows()
    if image_paths and total_time > 0:
        avg_ms = total_time / len(image_paths)
        print(f"\nAverage: {avg_ms:.1f}ms/image ({1000/avg_ms:.1f} FPS)")


if __name__ == "__main__":
    main()
