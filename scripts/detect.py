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
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model


def draw_boxes(image, results, class_names=None):
    """Draw detection boxes on image. Boxes expected in image-pixel space."""
    if class_names is None:
        class_names = [f"class_{i}" for i in range(80)]

    H, W = image.shape[:2]
    for det in results:
        boxes = det["boxes"]
        scores = det["scores"]
        labels = det["labels"]

        if boxes.numel() == 0:
            continue

        # boxes are in image-pixel space: [x1, y1, x2, y2]
        boxes_np = boxes.cpu().numpy()
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for box, score, label in zip(boxes_np, scores_np, labels_np):
            x1, y1, x2, y2 = box.astype(int)
            # Clamp to image bounds
            x1, x2 = max(0, min(W - 1, x1)), max(0, min(W - 1, x2))
            y1, y2 = max(0, min(H - 1, y1)), max(0, min(H - 1, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            cls_id = int(label)
            color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label_text = f"{class_names[cls_id]}: {score:.2f}"
            cv2.putText(image, label_text, (x1, max(y1 - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image


def load_image(path, size=640):
    """Load image, apply letterbox padding, return canvas + scale info."""
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Cannot load image: {path}")
    orig_h, orig_w = img.shape[:2]

    # Letterbox resize
    scale = min(size / orig_w, size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Canvas with letterbox padding
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # HWC -> CHW -> [0, 1]
    canvas_t = torch.from_numpy(canvas[:, :, ::-1]).permute(2, 0, 1).float() / 255.0

    return canvas_t, img, new_w, new_h, orig_w, orig_h, scale, pad_top, pad_left


def detect_image(model, image_path, device="mps", conf_thresh=0.25, iou_thresh=0.45, img_size=640):
    """Run detection on a single image."""
    model.eval()

    canvas_t, orig_img, new_w, new_h, orig_w, orig_h, scale, pad_top, pad_left = load_image(image_path, img_size)
    canvas_t = canvas_t.unsqueeze(0).to(device)

    with torch.no_grad():
        start = time.time()
        results = model.predict(canvas_t)
        latency = (time.time() - start) * 1000

    # PostProcess returns boxes in normalized canvas-space [0, 1].
    # Convert to canvas-pixel space, then clip padding, then scale to original image.
    for det in results:
        boxes = det["boxes"]  # (N, 4) normalized [0, 1]
        boxes[:, [0, 2]] *= img_size  # canvas pixel coords
        boxes[:, [1, 3]] *= img_size

        # Remove letterbox padding
        boxes[:, [0, 2]] -= pad_left
        boxes[:, [1, 3]] -= pad_top

        # Scale to original image
        boxes[:, [0, 2]] /= scale
        boxes[:, [1, 3]] /= scale

        # Clamp to original image dimensions
        boxes[:, 0].clamp_(0, orig_w)
        boxes[:, 2].clamp_(0, orig_w)
        boxes[:, 1].clamp_(0, orig_h)
        boxes[:, 3].clamp_(0, orig_h)

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
                       help="Path to pretrained weights")
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
        state = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "ema" in state and isinstance(state["ema"], dict):
                state = state["ema"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.info()

    class_names = None
    if args.class_names and os.path.exists(args.class_names):
        with open(args.class_names) as f:
            class_names = [l.strip() for l in f]

    save_dir = Path(args.save) if args.save else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Quick model test
    if args.source == "" or args.source == "0":
        print("Running quick model test...")
        x = torch.randn(1, 3, args.img_size, args.img_size).to(device)
        model.eval()
        with torch.no_grad():
            t0 = time.time()
            results = model.predict(x)
            t1 = time.time()
        print(f"Forward + postprocess: {(t1-t0)*1000:.1f}ms")
        print(f"Output: {len(results[0])} detections")
        return

    # Image or directory
    if os.path.isdir(args.source):
        image_paths = []
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG"]:
            image_paths.extend(Path(args.source).glob(ext))
        print(f"Found {len(image_paths)} images")
    else:
        image_paths = [Path(args.source)]

    total_time = 0
    for img_path in image_paths:
        try:
            result_img, _, latency = detect_image(
                model, str(img_path), device, args.conf, args.iou, args.img_size
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
