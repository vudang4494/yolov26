#!/usr/bin/env python3
"""
YOLOv26 Training Script with real dataset support.
"""
import argparse
import os
import sys
import time
import yaml
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model
from yolo26.core.loss import YOLOv26Loss
from yolo26.optimizers.musgd import MuSGD


def build_dataloader(data_yaml=None, batch_size=8, img_size=640, num_classes=80, shuffle=True, workers=4):
    """
    Build dataloader. If data_yaml is provided, loads real images from COCO-style YAML.
    Falls back to synthetic data if no YAML or images unavailable.
    """
    if data_yaml and os.path.exists(data_yaml):
        try:
            return _build_coco_dataloader(data_yaml, batch_size, img_size, num_classes, shuffle, workers)
        except Exception as e:
            print(f"[WARN] Failed to load {data_yaml}: {e}, using synthetic data")

    print("[INFO] Using synthetic dataset (no valid data.yaml)")
    return _build_synthetic_loader(batch_size, img_size, num_classes, shuffle)


def _build_coco_dataloader(data_yaml, batch_size, img_size, num_classes, shuffle, workers):
    """Load COCO-style dataset from YAML."""
    import cv2
    from torch.utils.data import Dataset, DataLoader

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    root = Path(data_yaml).parent / data.get("path", "")
    img_dir = root / data["train"] if "train" in data else root
    label_dir = root / (str(data["train"]).replace("images", "labels"))

    class COCODataset(Dataset):
        def __init__(self, img_dir, label_dir, img_size, num_classes):
            self.img_dir = Path(img_dir)
            self.label_dir = Path(label_dir)
            self.img_size = img_size
            self.nc = num_classes
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
                targets[:, 1] *= self.img_size / orig_w
                targets[:, 2] *= self.img_size / orig_h
                targets[:, 3] *= self.img_size / orig_w
                targets[:, 4] *= self.img_size / orig_h
                targets[:, 1:5] /= self.img_size
                targets[:, 1:5] = targets[:, 1:5].clamp(0, 1)
            else:
                targets = torch.zeros(0, 5)

            return img, targets

    dataset = COCODataset(img_dir, label_dir, img_size, num_classes)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"[INFO] Loaded {len(dataset)} images from {img_dir}")
    return loader


def _collate_fn(batch):
    """Collate batch: images + variable-length targets."""
    imgs = torch.stack([b[0] if isinstance(b[0], torch.Tensor) else torch.from_numpy(b[0]).permute(2, 0, 1).float() / 255.0
                        for b in batch])
    targets_list = []
    batch_idx = 0
    for img, targets in batch:
        if isinstance(targets, torch.Tensor) and targets.shape[0] > 0:
            t = targets.clone()
            t = torch.cat([torch.full((t.shape[0], 1), batch_idx, dtype=t.dtype, device=t.device), t], dim=1)
            targets_list.append(t)
        batch_idx += 1
    targets = torch.cat(targets_list, dim=0) if targets_list else torch.zeros(0, 6)
    return imgs, targets


def _build_synthetic_loader(batch_size, img_size, num_classes, shuffle):
    """Synthetic dataset for quick testing."""
    class DummyDataset:
        def __init__(self, size=1000):
            self.size = size

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


def train(args):
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, using CPU")
        device = "cpu"

    print(f"Training YOLOv26-{args.scale.upper()} on {device}")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch} | LR: {args.lr}")

    model = YOLOv26Model(
        num_classes=args.num_classes,
        scale=args.scale,
        image_size=args.img_size,
    ).to(device)

    model.info()
    criterion = YOLOv26Loss(num_classes=args.num_classes)

    optimizer = MuSGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    dataloader = build_dataloader(
        args.data or None,
        args.batch,
        args.img_size,
        args.num_classes,
        shuffle=True,
    )

    # AMP: use bfloat16 on MPS/CPU, float16 on CUDA
    use_amp = device in ("mps", "cuda")
    scaler = torch.amp.GradScaler(device, enabled=use_amp) if use_amp else None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device, dtype=torch.bfloat16 if device == "mps" else torch.float16):
                o2m, o2o = model(images)
                loss, losses = criterion(o2m, o2o, targets,
                                        model.anchor_list, args.img_size)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.epochs} | "
                      f"Batch {batch_idx}/{len(dataloader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Cls: {losses['cls']:.4f} | "
                      f"Box: {losses['box']:.4f}")

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - epoch_start

        print(f"\nEpoch {epoch+1}/{args.epochs} | "
              f"Avg Loss: {avg_loss:.4f} | "
              f"Time: {elapsed:.1f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}\n")

        if (epoch + 1) % args.save_every == 0:
            ckpt = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "loss": avg_loss,
                "scale": args.scale,
                "num_classes": args.num_classes,
            }
            ckpt_path = f"runs/yolo26_{args.scale}_e{epoch+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(ckpt, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_ckpt = {
        "epoch": args.epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scale": args.scale,
        "num_classes": args.num_classes,
    }
    final_path = f"runs/yolo26_{args.scale}_final.pt"
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    torch.save(final_ckpt, final_path)
    print(f"\nTraining complete. Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Training")
    parser.add_argument("--scale", type=str, default="n",
                       help="Model scale: n, s, m, l")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cpu, cuda")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--num-classes", type=int, default=80,
                       help="Number of classes")
    parser.add_argument("--data", type=str, default="",
                       help="Path to COCO-style data YAML")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
