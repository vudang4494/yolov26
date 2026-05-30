"""
Data augmentation for YOLOv26 training.
Implements: Mosaic, MixUp, Random Flip, Color Jitter, Random Crop.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import cv2


class MosaicAugmentation:
    """
    Mosaic: combine 4 images into 1.
    Randomly selects 3 other images from the batch and arranges them
    in a 2x2 grid. Targets are adjusted accordingly.
    """

    def __init__(self, img_size=640, prob=0.5):
        self.img_size = img_size
        self.prob = prob

    def __call__(self, images, targets):
        """
        Args:
            images: (B, C, H, W) tensor
            targets: (N, 6) tensor [batch_idx, cls, cx, cy, w, h] normalized
        Returns:
            mosaic_img: (C, H, W) tensor
            mosaic_targets: (M, 6) tensor
        """
        if random.random() > self.prob or images.shape[0] < 4:
            return images[0], targets[targets[:, 0] == 0]

        B, C, H, W = images.shape
        half = self.img_size // 2

        # Placeholder canvas
        mosaic_img = torch.zeros(C, self.img_size, self.img_size, device=images.device)
        mosaic_targets = []

        # Get 4 positions: top-left, top-right, bottom-left, bottom-right
        positions = [
            (0, 0, 0, half, 0, half),       # top-left: x1=0, y1=0, x2=half, y2=half
            (half, 0, half, self.img_size, 0, half),  # top-right
            (0, half, 0, half, half, self.img_size), # bottom-left
            (half, half, half, self.img_size, half, self.img_size),  # bottom-right
        ]

        indices = list(range(B))
        random.shuffle(indices)
        indices = indices[:4]

        for idx, (x1, y1, x2, y2) in zip(indices, positions):
            img = images[idx]  # (C, H, W)
            h, w = img.shape[1:]

            # Resize to half canvas
            resized = F.interpolate(
                img.unsqueeze(0), size=(half, half), mode="bilinear", align_corners=False
            ).squeeze(0)

            mosaic_img[:, y1:y2, x1:x2] = resized

            # Adjust targets
            tgt = targets[targets[:, 0] == idx]
            for t in tgt:
                cls, cx, cy, bw, bh = t[1].item(), t[2].item(), t[3].item(), t[4].item(), t[5].item()
                # Map to quadrant
                adj_cx = cx * 2 - (x1 / self.img_size)
                adj_cy = cy * 2 - (y1 / self.img_size)
                adj_bw = bw * 2
                adj_bh = bh * 2
                # Only keep targets within this quadrant
                if 0 <= adj_cx <= 1 and 0 <= adj_cy <= 1:
                    mosaic_targets.append([0, cls, adj_cx, adj_cy, adj_bw, adj_bh])

        if mosaic_targets:
            mosaic_targets = torch.tensor(mosaic_targets, device=images.device)
        else:
            mosaic_targets = torch.zeros(0, 6, device=images.device)

        return mosaic_img, mosaic_targets


class MixUpAugmentation:
    """
    MixUp: blend two images and their targets.
    targets_b = lambda * targets_a + (1-lambda) * targets_b
    """

    def __init__(self, alpha=0.5, prob=0.5):
        self.alpha = alpha
        self.prob = prob

    def __call__(self, img_a, tgt_a, img_b=None, tgt_b=None):
        """
        Args:
            img_a, img_b: (C, H, W) tensors
            tgt_a, tgt_b: (N, 6) tensors
        Returns:
            mixed_img, combined_targets
        """
        if random.random() > self.prob or img_b is None:
            return img_a, tgt_a

        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img_a + (1 - lam) * img_b

        combined = torch.cat([tgt_a, tgt_b], dim=0)
        return mixed_img, combined


class RandomFlip:
    """Random horizontal flip."""

    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, img, targets):
        if random.random() > self.prob:
            return img, targets

        img = torch.flip(img, dims=[2])  # flip W dimension
        if targets.shape[0] > 0:
            targets = targets.clone()
            targets[:, 2] = 1.0 - targets[:, 2]  # flip cx
        return img, targets


class ColorJitter:
    """Random color jittering: brightness, contrast, saturation, hue."""

    def __init__(self, brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, prob=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img: (C, H, W) tensor in [0, 1]
        Returns:
            (C, H, W) tensor
        """
        if random.random() > self.prob:
            return img

        C = img.shape[0]
        device = img.device

        # Brightness
        if self.brightness > 0:
            factor = 1.0 + random.uniform(-self.brightness, self.brightness)
            img = img * factor

        # Contrast
        if self.contrast > 0:
            factor = 1.0 + random.uniform(-self.contrast, self.contrast)
            mean = img.mean(dim=(1, 2), keepdim=True)
            img = (img - mean) * factor + mean

        # Saturation (only for 3-channel images)
        if self.saturation > 0 and C >= 3:
            factor = 1.0 + random.uniform(-self.saturation, self.saturation)
            gray = img[:3].mean(dim=0, keepdim=True)
            img = torch.cat([
                img[:3] * factor + gray * (1 - factor),
                img[3:]
            ], dim=0)
            img = img[:C]  # clip to original channels

        # Hue
        if self.hue > 0 and C >= 3:
            shift = random.uniform(-self.hue, self.hue)
            # Simple hue shift in HSV-like space
            t = torch.tensor([
                [shift * 0.5 + 0.5, 1.0, 1.0],
                [1.0, 1.0 - shift * 0.3, 1.0 + shift * 0.3],
                [1.0 + shift * 0.3, 1.0, 1.0 - shift * 0.3],
            ], device=device)
            rgb_shift = torch.tensor([
                [0.299, 0.587, 0.114],
                [-0.1687, -0.3312, 0.5],
                [0.5, -0.4187, -0.0813],
            ], device=device).T
            rgb_shift_inv = torch.inverse(rgb_shift)

            img_3 = img[:3]
            yuv = torch.einsum("ij,jhw->ihw", rgb_shift, img_3)
            yuv[1] += shift * 0.5
            yuv[2] += shift * 0.3
            img_3 = torch.einsum("ij,jhw->ihw", rgb_shift_inv, yuv)
            img = torch.cat([img_3.clamp(0, 1), img[3:]], dim=0)

        return img.clamp(0, 1)


class RandomCrop:
    """Random crop with IoU threshold — discards boxes with low overlap."""

    def __init__(self, img_size=640, min_iou=0.3, prob=0.3):
        self.img_size = img_size
        self.min_iou = min_iou
        self.prob = prob

    def __call__(self, img, targets):
        """
        Args:
            img: (C, H, W) tensor
            targets: (N, 6) tensor [batch_idx, cls, cx, cy, w, h]
        Returns:
            cropped_img, filtered_targets
        """
        if random.random() > self.prob or targets.shape[0] == 0:
            return img, targets

        C, H, W = img.shape
        scale = H / self.img_size

        # Random crop size between 0.5 and 1.0 of image
        crop_ratio = random.uniform(0.5, 1.0)
        crop_h = int(H * crop_ratio)
        crop_w = int(W * crop_ratio)

        # Random crop position
        top = random.randint(0, max(1, H - crop_h))
        left = random.randint(0, max(1, W - crop_w))

        img = img[:, top:top + crop_h, left:left + crop_w]
        img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)

        # Filter targets: keep only boxes with sufficient overlap
        kept = []
        for t in targets:
            cls, cx, cy, bw, bh = t[1].item(), t[2].item(), t[3].item(), t[4].item(), t[5].item()
            # Map to cropped coordinates
            cx_crop = (cx - left / W) / (crop_w / W)
            cy_crop = (cy - top / H) / (crop_h / H)

            if 0 <= cx_crop <= 1 and 0 <= cy_crop <= 1:
                kept.append([t[0].item(), cls, cx_crop, cy_crop, bw, bh])

        if kept:
            targets = torch.tensor(kept, device=targets.device)
        else:
            targets = torch.zeros(0, 6, device=targets.device)

        return img, targets


class AugmentationPipeline:
    """
    Full augmentation pipeline applied per image in a batch.
    Order: Mosaic -> RandomCrop -> RandomFlip -> ColorJitter -> MixUp
    """

    def __init__(self, img_size=640,
                 mosaic_prob=0.5, mixup_prob=0.3, flip_prob=0.5,
                 color_prob=0.5, crop_prob=0.3):
        self.img_size = img_size
        self.mosaic = MosaicAugmentation(img_size, prob=mosaic_prob)
        self.mixup = MixUpAugmentation(prob=mixup_prob)
        self.flip = RandomFlip(prob=flip_prob)
        self.color = ColorJitter(prob=color_prob)
        self.crop = RandomCrop(img_size, prob=crop_prob)

    def __call__(self, images, targets):
        """
        Args:
            images: (B, C, H, W) tensor
            targets: (N, 6) tensor [batch_idx, cls, cx, cy, w, h]
        Returns:
            aug_images: list of (C, H, W) tensors
            aug_targets: list of (M, 6) tensors
        """
        # Apply mosaic at batch level
        mosaic_img, mosaic_tgt = self.mosaic(images, targets)
        mosaic_img = mosaic_img.unsqueeze(0)  # (1, C, H, W)

        results = []
        aug_targets = []

        for i in range(mosaic_img.shape[0]):
            img = mosaic_img[i]
            tgt = mosaic_tgt

            # Random crop
            img, tgt = self.crop(img, tgt)

            # Random flip
            img, tgt = self.flip(img, tgt)

            # Color jitter
            img = self.color(img)

            results.append(img)
            aug_targets.append(tgt)

        return results, aug_targets
