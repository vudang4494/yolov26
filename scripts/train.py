#!/usr/bin/env python3
"""
YOLOv26 Training Script - Full pipeline with augmentation, EMA, warmup, validation.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model
from yolo26.core.loss import YOLOv26Loss
from yolo26.optimizers.musgd import MuSGD


# ─── EMA ────────────────────────────────────────────────────────────────────────

class ModelEMA:
    """
    Exponential Moving Average of model weights.
    Stabilizes training and typically improves final accuracy by 0.5-1% mAP.
    """

    def __init__(self, model, decay=0.9999, device="cpu"):
        import copy
        self.ema = copy.deepcopy(model).eval()
        self.ema.to(device)
        self.decay = decay
        self.device = device
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_p, p in zip(self.ema.parameters(), model.parameters()):
                ema_p.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


# ─── Data augmentation ────────────────────────────────────────────────────────

from yolo26.augmentation import (
    MosaicAugmentation, MixUpAugmentation, RandomFlip,
    ColorJitter, RandomCrop, AugmentationPipeline,
)


# ─── Data loading ─────────────────────────────────────────────────────────────

def build_dataloader(data_yaml=None, batch_size=8, img_size=640, num_classes=80,
                     shuffle=True, workers=4, augment=False):
    if data_yaml and os.path.exists(data_yaml):
        try:
            return _build_coco_dataloader(data_yaml, batch_size, img_size,
                                          num_classes, shuffle, workers, augment)
        except Exception as e:
            print(f"[WARN] Failed to load {data_yaml}: {e}, using synthetic data")

    print("[INFO] Using synthetic dataset")
    return _build_synthetic_loader(batch_size, img_size, num_classes, shuffle, augment)


def _build_coco_dataloader(data_yaml, batch_size, img_size, num_classes,
                           shuffle, workers, augment):
    import cv2
    from torch.utils.data import Dataset, DataLoader

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    root = Path(data_yaml).parent / data.get("path", "")
    img_dir = root / (data["train"] if "train" in data else "images/train")
    label_dir = root / (str(data["train"]).replace("images", "labels"))

    class COCODataset(Dataset):
        def __init__(self, img_dir, label_dir, img_size, num_classes, augment=False):
            self.img_dir = Path(img_dir)
            self.label_dir = Path(label_dir)
            self.img_size = img_size
            self.nc = num_classes
            self.augment = augment
            self.flip = RandomFlip(prob=0.5)
            self.color = ColorJitter(prob=0.5)
            self.img_files = sorted([
                f for f in self.img_dir.iterdir()
                if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
            ])
            if not self.img_files:
                raise FileNotFoundError(f"No images in {self.img_dir}")

        def __len__(self):
            return len(self.img_files)

        def __getitem__(self, idx):
            img_path = self.img_files[idx]
            label_path = self.label_dir / (img_path.stem + ".txt")

            img = cv2.imread(str(img_path))
            if img is None:
                img = torch.zeros(3, self.img_size, self.img_size)
                targets = torch.zeros(0, 6)
                return img, targets

            orig_h, orig_w = img.shape[:2]
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = torch.from_numpy(img[:, :, ::-1]).permute(2, 0, 1).float() / 255.0

            targets = []
            if label_path.exists():
                with open(label_path) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        targets.append([cls, cx, cy, w, h])

            if targets:
                targets = torch.tensor(targets)
            else:
                targets = torch.zeros(0, 5)

            # Apply augmentation
            if self.augment and targets.shape[0] > 0:
                targets_full = torch.cat([
                    torch.zeros(targets.shape[0], 1), targets
                ], dim=1)
                img, targets_full = self.flip(img, targets_full)
                img = self.color(img)
                targets = targets_full[:, 1:]

            return img, targets

    dataset = COCODataset(img_dir, label_dir, img_size, num_classes, augment)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=workers, collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[INFO] Loaded {len(dataset)} images from {img_dir} (augment={augment})")
    return loader


def _collate_fn(batch):
    imgs = torch.stack([b[0] if isinstance(b[0], torch.Tensor) else
                        torch.from_numpy(b[0]).permute(2, 0, 1).float() / 255.0
                        for b in batch])
    targets_list = []
    batch_idx = 0
    for img, targets in batch:
        if isinstance(targets, torch.Tensor) and targets.shape[0] > 0:
            t = targets.clone()
            t = torch.cat([torch.full((t.shape[0], 1), batch_idx,
                       dtype=t.dtype, device=t.device), t], dim=1)
            targets_list.append(t)
        batch_idx += 1
    targets = torch.cat(targets_list, dim=0) if targets_list else torch.zeros(0, 6)
    return imgs, targets


def _build_synthetic_loader(batch_size, img_size, num_classes, shuffle, augment):
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size
            self.flip = RandomFlip(prob=0.5)
            self.color = ColorJitter(prob=0.5)

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            img = torch.rand(3, img_size, img_size)
            num_gt = torch.randint(1, 10, (1,)).item()
            boxes = torch.rand(num_gt, 4)
            boxes[:, 2:] *= 0.2
            boxes[:, 0] = boxes[:, 0].clamp(boxes[:, 2] / 2, 1 - boxes[:, 2] / 2)
            boxes[:, 1] = boxes[:, 1].clamp(boxes[:, 3] / 2, 1 - boxes[:, 3] / 2)
            cls = torch.randint(0, num_classes, (num_gt,))
            targets = torch.cat([cls.float().unsqueeze(1), boxes], dim=1)

            if augment:
                img = self.color(img)
                targets_full = torch.cat([torch.zeros(num_gt, 1), targets], dim=1)
                img, targets_full = self.flip(img, targets_full)
                targets = targets_full[:, 1:]

            return img, targets

    class DummyLoader:
        def __init__(self, ds, bs, shuffle):
            self.ds = ds
            self.bs = bs
            self.shuffle = shuffle

        def __iter__(self):
            indices = torch.randperm(len(self.ds)).tolist() if self.shuffle else list(range(len(self.ds)))
            for i in range(0, len(indices), self.bs):
                batch = [self.ds[indices[j]] for j in range(i, min(i + self.bs, len(indices)))]
                imgs = torch.stack([b[0] for b in batch])
                targets_list = []
                for bi, (_, tgt) in enumerate(batch):
                    if tgt.shape[0] > 0:
                        t = torch.cat([torch.full((tgt.shape[0], 1), bi, dtype=tgt.dtype), tgt], dim=1)
                        targets_list.append(t)
                targets = torch.cat(targets_list, dim=0) if targets_list else torch.zeros(0, 6)
                yield imgs, targets

        def __len__(self):
            return (len(self.ds) + self.bs - 1) // self.bs

    return DummyLoader(DummyDataset(), batch_size, shuffle)


# ─── Validation / mAP ──────────────────────────────────────────────────────────

def box_iou(boxes1, boxes2, eps=1e-7):
    """IoU between two sets of boxes in cxcywh format."""
    if boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return torch.zeros(boxes1.shape[0], boxes2.shape[0])
    # Convert cxcywh -> xyxy
    b1x1 = boxes1[..., 0] - boxes1[..., 2] / 2
    b1y1 = boxes1[..., 1] - boxes1[..., 3] / 2
    b1x2 = boxes1[..., 0] + boxes1[..., 2] / 2
    b1y2 = boxes1[..., 1] + boxes1[..., 3] / 2
    area1 = (b1x2 - b1x1).clamp(min=0) * (b1y2 - b1y1).clamp(min=0)

    b2x1 = boxes2[..., 0] - boxes2[..., 2] / 2
    b2y1 = boxes2[..., 1] - boxes2[..., 3] / 2
    b2x2 = boxes2[..., 0] + boxes2[..., 2] / 2
    b2y2 = boxes2[..., 1] + boxes2[..., 3] / 2
    area2 = (b2x2 - b2x1).clamp(min=0) * (b2y2 - b2y1).clamp(min=0)

    inter_x1 = torch.max(b1x1.unsqueeze(1), b2x1.unsqueeze(0))
    inter_y1 = torch.max(b1y1.unsqueeze(1), b2y1.unsqueeze(0))
    inter_x2 = torch.min(b1x2.unsqueeze(1), b2x2.unsqueeze(0))
    inter_y2 = torch.min(b1y2.unsqueeze(1), b2y2.unsqueeze(0))
    inter = (inter_x2 - inter_x1).clamp(min=0) * (inter_y2 - inter_y1).clamp(min=0)
    return inter / (area1.unsqueeze(1) + area2.unsqueeze(0) - inter + eps)


def compute_ap(precisions, recalls):
    """11-point interpolation AP."""
    if len(recalls) == 0:
        return 0.0
    recalls = sorted(set(recalls + [0.0, 1.0]))
    precisions = sorted(set(precisions + [0.0]))
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    ap = sum((recalls[i + 1] - recalls[i]) * precisions[i + 1]
             for i in range(len(recalls) - 1))
    return ap


def validate(model, val_loader, device, num_classes=80, image_size=640, max_batches=100):
    """Run validation: compute mAP@0.5."""
    model.eval()
    all_preds = []
    all_gts = []
    batch_count = 0

    with torch.no_grad():
        for images, targets in val_loader:
            if batch_count >= max_batches:
                break
            images = images.to(device)
            results = model.predict(images)
            for b, det in enumerate(results):
                gt_b = targets[targets[:, 0] == b]
                all_gts.append(gt_b)
                all_preds.append(det)
            batch_count += 1

    # Compute AP per class
    aps = []
    for cls in range(num_classes):
        cls_dets = [(i, p) for i, p in enumerate(all_preds)
                    for j in range(len(p["labels"]))
                    if p["labels"][j] == cls]
        cls_gts = [(i, g) for i, g in enumerate(all_gts)
                   for j in range(len(g))
                   if int(g[j, 1]) == cls]

        if not cls_gts:
            continue

        cls_dets.sort(key=lambda x: x[1]["scores"].cpu(), reverse=True)
        tp = torch.zeros(len(cls_dets))
        fp = torch.zeros(len(cls_dets))
        gt_matched = [False] * len(cls_gts)

        for k, (img_idx, det) in enumerate(cls_dets):
            best_iou = 0
            best_gt = -1
            # det["boxes"] is xyxy normalized [0,1]; gt is (batch_idx, cls, cx, cy, w, h) normalized [0,1]
            # Both in canvas-space, so convert xyxy -> cxcywh for IoU
            det_boxes = det["boxes"].cpu()
            det_cx = (det_boxes[:, 0] + det_boxes[:, 2]) / 2
            det_cy = (det_boxes[:, 1] + det_boxes[:, 3]) / 2
            det_w = det_boxes[:, 2] - det_boxes[:, 0]
            det_h = det_boxes[:, 3] - det_boxes[:, 1]
            det_cxcywh = torch.stack([det_cx, det_cy, det_w, det_h], dim=1)

            for gt_idx, (gt_img_idx, gt) in enumerate(cls_gts):
                if gt_matched[gt_idx] or gt_img_idx != img_idx:
                    continue
                # gt = (batch_idx, cls, cx, cy, w, h) normalized [0,1]
                gt_cxcywh = gt[2:].unsqueeze(0)  # (1, 4) cxcywh
                iou = box_iou(det_cxcywh, gt_cxcywh)[0, 0].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gt_idx
            if best_iou >= 0.5 and best_gt >= 0:
                tp[k] = 1
                gt_matched[best_gt] = True
            else:
                fp[k] = 1

        tp_cum = tp.cumsum(dim=0)
        fp_cum = fp.cumsum(dim=0)
        recalls = tp_cum / len(cls_gts)
        precisions = tp_cum / (tp_cum + fp_cum + 1e-9)
        ap = compute_ap(precisions.tolist(), recalls.tolist())
        aps.append(ap)

    mAP = (sum(aps) / len(aps) * 100) if aps else 0.0
    model.train()
    return {"mAP@0.5": round(mAP, 2), "n_classes": len(aps)}


# ─── Training ──────────────────────────────────────────────────────────────────

def train(args):
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    print(f"\n{'='*60}")
    print(f"  YOLOv26-{args.scale.upper()} Training on {device}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")
    print(f"  Warmup: {args.warmup_epochs} | EMA decay: {args.ema_decay}")
    print(f"  Data: {args.data or 'synthetic'}")
    print(f"{'='*60}\n")

    # Build model
    model = YOLOv26Model(
        num_classes=args.num_classes,
        scale=args.scale,
        image_size=args.img_size,
    ).to(device)
    model.info()

    # EMA
    ema = ModelEMA(model, decay=args.ema_decay, device=device)

    # Loss
    criterion = YOLOv26Loss(
        num_classes=args.num_classes,
        alpha_prog=args.alpha_prog,
        beta_stal=args.beta_stal,
        o2o_weight=args.o2o_weight,
        box_weight=args.box_weight,
    )

    # Optimizer
    optimizer = MuSGD(
        model.parameters(), lr=args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay,
        nesterov=True, ns_warmup=args.warmup_epochs * 100,
    )

    # Warmup + CosineAnnealing
    total_steps = args.epochs
    warmup_steps = args.warmup_epochs

    def lr_fn(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_fn)

    # Dataloaders
    train_loader = build_dataloader(
        args.data, args.batch, args.img_size, args.num_classes,
        shuffle=True, workers=args.workers, augment=True,
    )
    val_loader = build_dataloader(
        args.data, args.batch, args.img_size, args.num_classes,
        shuffle=False, workers=args.workers, augment=False,
    ) if args.data else None

    # AMP
    use_amp = device in ("mps", "cuda")
    scaler = torch.amp.GradScaler(device, enabled=use_amp) if use_amp else None

    # Logging
    os.makedirs(args.save_dir, exist_ok=True)

    global_step = 0
    best_map = 0.0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Skip empty batches
            if targets.shape[0] == 0:
                continue

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device,
                                    dtype=torch.bfloat16 if device == "mps" else torch.float16,
                                    enabled=use_amp):
                o2m, o2o = model(images)
                loss, losses = criterion(o2m, o2o, targets,
                                        model.anchor_list, args.img_size)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            scheduler.step()
            ema.update(model)

            global_step += 1
            epoch_loss += loss.item()

            lr_now = scheduler.get_last_lr()[0]
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "cls": f"{losses['cls'].item():.4f}",
                "box": f"{losses['box'].item():.4f}",
                "lr": f"{lr_now:.6f}",
            })

        avg_loss = epoch_loss / max(len(train_loader), 1)
        elapsed = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        # Validation
        val_map = 0.0
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            val_results = validate(model, val_loader, device,
                                  args.num_classes, args.img_size, max_batches=100)
            val_map = val_results["mAP@0.5"]
            print(f"\n  Epoch {epoch+1} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Val mAP@0.5: {val_map:.2f}% | "
                  f"Time: {elapsed:.1f}s | "
                  f"LR: {current_lr:.6f}")
            if val_map > best_map:
                best_map = val_map
                torch.save(
                    ema.ema.state_dict(),
                    os.path.join(args.save_dir, f"yolo26_{args.scale}_best.pt")
                )
                print(f"  New best mAP: {best_map:.2f}%")
        else:
            print(f"\n  Epoch {epoch+1} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {elapsed:.1f}s | "
                  f"LR: {current_lr:.6f}")

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "ema": ema.ema.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
                "val_map": val_map,
                "best_map": best_map,
                "scale": args.scale,
                "num_classes": args.num_classes,
            }
            ckpt_path = os.path.join(args.save_dir,
                                     f"yolo26_{args.scale}_e{epoch+1}.pt")
            torch.save(ckpt, ckpt_path)
            print(f"  Saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(args.save_dir, f"yolo26_{args.scale}_final.pt")
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "ema": ema.ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scale": args.scale,
        "num_classes": args.num_classes,
        "best_map": best_map,
    }, final_path)
    print(f"\nTraining complete. Best mAP@0.5: {best_map:.2f}%")
    print(f"Model saved to {final_path}")


# ─── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Training")
    parser.add_argument("--scale", type=str, default="n",
                       help="Model scale: n, s, m, l")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--num-classes", type=int, default=80)
    parser.add_argument("--data", type=str, default="",
                       help="Path to COCO-style data YAML")
    parser.add_argument("--save-dir", type=str, default="runs")
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--val-every", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--alpha-prog", type=float, default=0.3,
                       help="ProgLoss class re-weighting strength")
    parser.add_argument("--beta-stal", type=float, default=0.3,
                       help="STAL small-target weight boost")
    parser.add_argument("--o2o-weight", type=float, default=1.0,
                       help="O2O branch loss weight")
    parser.add_argument("--box-weight", type=float, default=7.5,
                       help="Box regression loss weight")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
