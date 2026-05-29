#!/usr/bin/env python3
"""
YOLOv26 Training Script.
"""
import argparse
import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from yolo26.core.model import YOLOv26Model
from yolo26.core.loss import YOLOv26Loss
from yolo26.optimizers.musgd import MuSGD


def build_dataloader(data_yaml, batch_size, img_size, shuffle=True):
    """Build a simple synthetic dataloader for testing."""
    class DummyDataset:
        def __init__(self, size=1000, img_size=640, num_classes=80):
            self.size = size
            self.img_size = img_size
            self.nc = num_classes

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            img = torch.randn(3, self.img_size, self.img_size)
            num_gt = torch.randint(1, 10, (1,)).item()
            boxes = []
            for _ in range(num_gt):
                x, y = torch.rand(2)
                w = torch.rand(1) * 0.2
                h = torch.rand(1) * 0.2
                boxes.append([x, y, w, h])
            boxes = torch.tensor(boxes) if boxes else torch.zeros(0, 4)
            cls = torch.randint(0, self.nc, (num_gt,))
            targets = torch.zeros(num_gt, 6)
            targets[:, 1] = cls.float()
            targets[:, 2:] = boxes
            return img, targets

    class DummyLoader:
        def __init__(self, dataset, batch_size, shuffle):
            self.dataset = dataset
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = list(range(len(dataset)))

        def __iter__(self):
            if self.shuffle:
                indices = torch.randperm(len(self.dataset)).tolist()
            else:
                indices = self.indices
            for i in range(0, len(indices), self.batch_size):
                batch_idx = indices[i:i + self.batch_size]
                imgs = []
                targets_list = []
                for idx in batch_idx:
                    img, targets = self.dataset[idx]
                    targets[:, 0] = idx
                    imgs.append(img)
                    targets_list.append(targets)
                yield torch.stack(imgs), torch.cat(targets_list, 0)

        def __len__(self):
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    dataset = DummyDataset(size=1000, img_size=img_size)
    return DummyLoader(dataset, batch_size, shuffle)


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

    dataloader = build_dataloader(None, args.batch, args.img_size)

    scaler = torch.amp.GradScaler("cuda") if device == "cuda" else None

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_idx, (images, targets) in enumerate(dataloader):
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.amp.autocast("cuda"):
                    o2m, o2o = model(images)
                    loss, losses = criterion(o2m, o2o, targets,
                                            model.anchor_list, args.img_size)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                o2m, o2o = model(images)
                loss, losses = criterion(o2m, o2o, targets,
                                        model.anchor_list, args.img_size)
                loss.backward()
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
            ckpt_path = f"runs/yolo26_{args.scale}_e{epoch+1}.pt"
            os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    final_path = f"runs/yolo26_{args.scale}_final.pt"
    os.makedirs(os.path.dirname(final_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Model saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(description="YOLOv26 Training")
    parser.add_argument("--scale", type=str, default="n",
                       help="Model scale: n, s, m, l, x")
    parser.add_argument("--epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8,
                       help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                       help="Learning rate")
    parser.add_argument("--device", type=str, default="mps",
                       help="Device: mps, cpu, cuda")
    parser.add_argument("--img-size", type=int, default=640,
                       help="Input image size")
    parser.add_argument("--num-classes", type=int, default=80,
                       help="Number of classes")
    parser.add_argument("--data", type=str, default="",
                       help="Path to data YAML (optional)")
    parser.add_argument("--save-every", type=int, default=10,
                       help="Save checkpoint every N epochs")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
