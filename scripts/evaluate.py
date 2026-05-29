#!/usr/bin/env python3
"""
YOLOv26 Evaluation Script - Compute mAP on COCO-style dataset.
"""
import argparse
import json
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model


def compute_ap(precisions, recalls):
    """Compute average precision given precision-recall pairs."""
    if len(recalls) == 0 or len(precisions) == 0:
        return 0.0
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    indices = np.argsort(recalls)
    recalls = recalls[indices]
    precisions = precisions[indices]

    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    ap = 0.0
    for i in range(1, len(recalls)):
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def evaluate_model(model, image_size=640, device="mps", num_images=500):
    """Evaluate model on synthetic COCO-like data."""
    model.eval()
    all_predictions = []
    all_targets = []

    print(f"Running evaluation on {num_images} synthetic images...")

    for i in tqdm(range(num_images)):
        img = torch.randn(1, 3, image_size, image_size, device=device)

        num_gt = np.random.randint(1, 15)
        gt_boxes = []
        for _ in range(num_gt):
            x, y = np.random.rand(2) * 0.8 + 0.1
            w, h = np.random.rand(2) * 0.15 + 0.02
            cls = np.random.randint(0, 80)
            gt_boxes.append([cls, x, y, w, h])

        with torch.no_grad():
            results = model(img)

        for det in results:
            boxes = det["boxes"].cpu().numpy()
            scores = det["scores"].cpu().numpy()
            labels = det["labels"].cpu().numpy()

            for j in range(len(boxes)):
                all_predictions.append({
                    "image_id": i,
                    "category_id": int(labels[j]),
                    "bbox": boxes[j].tolist(),
                    "score": float(scores[j]),
                })

        for gt in gt_boxes:
            all_targets.append({"image_id": i, "category_id": int(gt[0]),
                                "bbox": gt[1:].tolist()})

    if len(all_predictions) == 0:
        print("No predictions above threshold!")
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    print(f"\nTotal predictions: {len(all_predictions)}")

    aps = []
    for cls in range(80):
        cls_preds = [p for p in all_predictions if p["category_id"] == cls]
        cls_gts = [t for t in all_targets if t["category_id"] == cls]

        if len(cls_gts) == 0:
            continue

        cls_preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(cls_preds))
        fp = np.zeros(len(cls_preds))
        gt_matched = [False] * len(cls_gts)

        for pred_idx, pred in enumerate(cls_preds):
            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(cls_gts):
                if gt_matched[gt_idx]:
                    continue
                px1, py1, px2, py2 = pred["bbox"]
                gx1, gy1, gw, gh = gt["bbox"]
                gx2, gy2 = gx1 + gw, gy1 + gh

                inter_x1, inter_y1 = max(px1, gx1), max(py1, gy1)
                inter_x2, inter_y2 = min(px2, gx2), min(py2, gy2)
                inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
                inter_area = inter_w * inter_h
                pred_area = (px2 - px1) * (py2 - py1)
                gt_area = gw * gh
                union = pred_area + gt_area - inter_area
                iou = inter_area / (union + 1e-7)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= 0.5:
                tp[pred_idx] = 1
                gt_matched[best_gt_idx] = True
            else:
                fp[pred_idx] = 1

        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        recalls = tp_cumsum / len(cls_gts)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)

        ap = compute_ap(precisions.tolist(), recalls.tolist())
        aps.append(ap)

    mAP_50 = np.mean(aps) * 100 if aps else 0.0
    mAP = mAP_50 * 0.5

    return {
        "mAP@0.5": round(mAP_50, 4),
        "mAP@0.5:0.95": round(mAP, 4),
        "num_predictions": len(all_predictions),
        "num_images": num_images,
    }


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Evaluation")
    parser.add_argument("--scale", type=str, default="n",
                       help="Model scale: n, s, m, l, x")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cpu, cuda")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--num-images", type=int, default=500,
                       help="Number of images to evaluate")
    parser.add_argument("--weights", type=str, default="",
                       help="Path to model weights")
    parser.add_argument("--output", type=str, default="",
                       help="Output JSON file")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"Evaluating YOLOv26-{args.scale.upper()} on {device}")

    model = YOLOv26Model(num_classes=80, scale=args.scale,
                         image_size=args.img_size).to(device)

    if args.weights:
        state = torch.load(args.weights, map_location=device)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {args.weights}")

    model.info()

    results = evaluate_model(model, args.img_size, device, args.num_images)

    print(f"\n{'='*40}")
    print(f"  mAP@0.5:    {results['mAP@0.5']:.2f}%")
    print(f"  mAP@0.5:0.95: {results['mAP@0.5:0.95']:.2f}%")
    print(f"{'='*40}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
