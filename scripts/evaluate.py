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


def box_iou(boxes1, boxes2, eps=1e-7):
    """IoU between two sets of cxcywh boxes."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])
    inter = torch.min(boxes1[..., 2:], boxes2[..., 2:]).clamp(min=0).prod(dim=-1)
    area1 = (boxes1[..., 2] * boxes1[..., 3]).unsqueeze(1)
    area2 = (boxes2[..., 2] * boxes2[..., 3]).unsqueeze(0)
    return inter / (area1 + area2 - inter + eps)


def compute_ap(precisions, recalls):
    """11-point interpolation AP."""
    if len(recalls) == 0:
        return 0.0
    recalls = sorted(set(recalls + [0.0, 1.0]))
    precisions = sorted(set(precisions + [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    return sum((recalls[i + 1] - recalls[i]) * precisions[i + 1]
               for i in range(len(recalls) - 1))


def compute_map_recall(precision, recall):
    """VOC-style AP computation with recall interpolation."""
    mrec = np.concatenate([[0.0], recall, [1.0]])
    mpre = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_model(model, data_yaml=None, image_size=640, device="mps",
                  num_images=500, conf_thresh=0.001):
    """
    Evaluate model on COCO-style dataset (real or synthetic fallback).
    Uses COCO mAP@0.5:0.95 computation.
    """
    model.eval()

    # Load real data if available
    all_predictions = []
    all_targets = []
    image_ids = []

    if data_yaml and os.path.exists(data_yaml):
        all_predictions, all_targets, image_ids = _load_coco_annotations(
            data_yaml, model, image_size, device, num_images, conf_thresh
        )
    else:
        print(f"Running evaluation on {num_images} synthetic images...")
        from yolo26.core.postprocess import PostProcess
        eval_postprocess = PostProcess(reg_max=4, num_classes=80, conf_thresh=0.0)
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
                o2m, o2o = model.forward(img)
                # Bypass internal conf_thresh by calling postprocess directly
                results = eval_postprocess(o2o, model.anchor_list,
                                          [8, 16, 32], image_size=image_size, mode="o2o")
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
                all_targets.append({
                    "image_id": i, "category_id": int(gt[0]),
                    "bbox": [float(v) for v in gt[1:]]
                })
            image_ids.append(i)

    if len(all_predictions) == 0:
        print("No predictions above threshold!")
        return {"mAP@0.5": 0.0, "mAP@0.5:0.95": 0.0}

    # COCO-style mAP computation across IoU thresholds
    iou_thresholds = np.linspace(0.5, 0.95, 10)
    map_results = {}

    for iou_thresh in iou_thresholds:
        aps = []
        for cls in range(80):
            cls_preds = [p for p in all_predictions if p["category_id"] == cls]
            cls_gts = [t for t in all_targets if t["category_id"] == cls]

            if not cls_gts:
                continue

            cls_preds.sort(key=lambda x: x["score"], reverse=True)
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            gt_matched = [False] * len(cls_gts)

            for pred_idx, pred in enumerate(cls_preds):
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(cls_gts):
                    if gt_matched[gt_idx] or gt["image_id"] != pred["image_id"]:
                        continue
                    px1, py1, px2, py2 = pred["bbox"]
                    gx1, gy1, gw, gh = gt["bbox"]
                    gx2, gy2 = gx1 + gw, gy1 + gh
                    inter_x1 = max(px1, gx1)
                    inter_y1 = max(py1, gy1)
                    inter_x2 = min(px2, gx2)
                    inter_y2 = min(py2, gy2)
                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h
                    pred_area = (px2 - px1) * (py2 - py1)
                    gt_area = gw * gh
                    union = pred_area + gt_area - inter_area
                    iou = inter_area / (union + 1e-7)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

                if best_iou >= iou_thresh:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / len(cls_gts)
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-9)
            ap = compute_map_recall(precisions, recalls)
            aps.append(ap)

        map_key = f"AP@IoU={iou_thresh:.2f}"
        map_results[map_key] = round(np.mean(aps) * 100, 2) if aps else 0.0

    # Overall mAP
    overall_map = np.mean(list(map_results.values()))
    map_results["mAP@0.5"] = map_results.get("AP@IoU=0.50", 0.0)
    map_results["mAP@0.5:0.95"] = round(overall_map, 2)
    map_results["num_predictions"] = len(all_predictions)
    map_results["num_images"] = len(set(image_ids))

    return map_results


def _load_coco_annotations(data_yaml, model, image_size, device, num_images, conf_thresh):
    """Load real COCO val images and evaluate."""
    import yaml
    import cv2

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    root = Path(data_yaml).parent / data.get("path", "")
    val_img_dir = root / (data.get("val", "val2017"))
    val_lbl_dir = root / (str(data.get("val", "val2017")).replace("images", "labels"))

    img_files = sorted([
        f for f in val_img_dir.iterdir()
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
    ])[:num_images]

    print(f"Evaluating {len(img_files)} real images from {val_img_dir}")
    all_predictions = []
    all_targets = []
    image_ids = []

    for img_path in tqdm(img_files):
        img_id = img_path.stem
        image_ids.append(img_id)

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (image_size, image_size))
        img_t = torch.from_numpy(img_resized[:, :, ::-1]).permute(2, 0, 1).float() / 255.0
        img_t = img_t.unsqueeze(0).to(device)

        # Predict
        with torch.no_grad():
            results = model.predict(img_t)

        for det in results:
            boxes = det["boxes"].cpu()
            scores = det["scores"].cpu()
            labels = det["labels"].cpu()

            for j in range(len(boxes)):
                if scores[j] < conf_thresh:
                    continue
                # Convert xyxy back to original scale for IoU
                box = boxes[j].clone()
                box[[0, 2]] *= orig_w / image_size
                box[[1, 3]] *= orig_h / image_size
                all_predictions.append({
                    "image_id": img_id,
                    "category_id": int(labels[j]),
                    "bbox": box.tolist(),
                    "score": scores[j].item(),
                })

        # Load GT
        label_path = val_lbl_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])
                    all_targets.append({
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [cx * orig_w, cy * orig_h, w * orig_w, h * orig_h],
                    })

    return all_predictions, all_targets, image_ids


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Evaluation")
    parser.add_argument("--scale", type=str, default="n")
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-images", type=int, default=500)
    parser.add_argument("--data", type=str, default="",
                       help="Path to COCO-style data.yaml (real evaluation)")
    parser.add_argument("--weights", type=str, default="",
                       help="Path to model weights")
    parser.add_argument("--output", type=str, default="",
                       help="Output JSON file")
    parser.add_argument("--conf-thresh", type=float, default=0.001,
                       help="Confidence threshold for evaluation")
    args = parser.parse_args()

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"Evaluating YOLOv26-{args.scale.upper()} on {device}")

    model = YOLOv26Model(num_classes=80, scale=args.scale,
                         image_size=args.img_size,
                         conf_thresh=args.conf_thresh).to(device)

    if args.weights:
        state = torch.load(args.weights, map_location=device, weights_only=False)
        if isinstance(state, dict):
            if "ema" in state:
                state = state["ema"]
            elif "model" in state and isinstance(state["model"], dict):
                state = state["model"]
            elif "state_dict" in state:
                state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {args.weights}")

    model.info()

    results = evaluate_model(model, args.data or None, args.img_size,
                            device, args.num_images, args.conf_thresh)

    print(f"\n{'='*40}")
    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}%")
        else:
            print(f"  {k}: {v}")
    print(f"{'='*40}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
