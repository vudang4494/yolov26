import torch
import torch.nn as nn
import math
from .backbone import YOLOv26Backbone
from .neck import YOLOv26Neck
from .head import YOLOv26Head
from .postprocess import PostProcess


SCALE_CONFIGS = {
    "n": {
        "channels": (64, 128, 256, 512, 1024),
        "depths": (1, 2, 4, 4, 2),
        "width_mult": 0.25,
        "num_repeats": (1, 3, 6, 6, 3),
    },
    "s": {
        "channels": (64, 128, 256, 512, 1024),
        "depths": (1, 3, 6, 6, 3),
        "width_mult": 0.5,
        "num_repeats": (1, 3, 9, 9, 3),
    },
    "m": {
        "channels": (80, 160, 320, 640, 1280),
        "depths": (2, 4, 8, 8, 4),
        "width_mult": 0.75,
        "num_repeats": (2, 6, 12, 12, 4),
    },
    "l": {
        "channels": (96, 192, 384, 768, 1536),
        "depths": (3, 6, 12, 12, 6),
        "width_mult": 1.0,
        "num_repeats": (3, 9, 18, 18, 6),
    },
    "x": {
        "channels": (112, 224, 448, 896, 1792),
        "depths": (4, 8, 16, 16, 8),
        "width_mult": 1.25,
        "num_repeats": (4, 12, 24, 24, 8),
    },
}


def generate_anchors(feature_sizes, strides, image_size=640, device="cpu"):
    """
    Generate anchor boxes for each feature level.
    Returns list of (n_anchors, 4) tensors: [cx, cy, w, h] normalized to [0, 1].
    """
    anchors = []
    for (h, w), stride in zip(feature_sizes, strides):
        shifts_x = (torch.arange(w, device=device) + 0.5) * stride / image_size
        shifts_y = (torch.arange(h, device=device) + 0.5) * stride / image_size
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)

        anchor_w = stride / image_size
        anchor_h = stride / image_size

        anchor = torch.stack([shift_x, shift_y,
                              torch.full_like(shift_x, anchor_w),
                              torch.full_like(shift_x, anchor_h)], dim=1)
        anchors.append(anchor)
    return anchors


class YOLOv26Model(nn.Module):
    """
    YOLOv26: Pure PyTorch object detector with 5 innovations.

    Architecture:
        - Backbone: CSPDarknet + RefinedC3k2 + SPPF
        - Neck: PAN-FPN
        - Head: Dual Head (O2M + O2O)

    Args:
        num_classes: Number of object classes (80 for COCO)
        scale: Model scale 'n', 's', 'm', 'l', or 'x'
        image_size: Input image size (default 640)
        reg_max: DFL removal - regression channels (default 4)
        conf_thresh: Confidence threshold
        iou_thresh: IoU threshold for NMS (O2M branch)
    """

    def __init__(self, num_classes=80, scale="n", image_size=640,
                 reg_max=4, conf_thresh=0.25, iou_thresh=0.45):
        super().__init__()
        self.nc = num_classes
        self.scale_name = scale
        self.image_size = image_size
        self.reg_max = reg_max
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        if scale not in SCALE_CONFIGS:
            raise ValueError(f"Scale must be one of {list(SCALE_CONFIGS.keys())}")
        cfg = SCALE_CONFIGS[scale]

        self.backbone = YOLOv26Backbone(
            in_ch=3,
            channels=cfg["channels"],
            depths=cfg["depths"],
            width_mult=cfg["width_mult"],
        )

        wm = cfg["width_mult"]
        backbone_out_chs = [int(c * wm) for c in cfg["channels"][2:]]

        self.neck = YOLOv26Neck(
            in_chs=backbone_out_chs,
            out_ch=int(256 * wm),
            width_mult=wm,
        )

        self.head = YOLOv26Head(
            num_classes=num_classes,
            in_ch=int(256 * wm),
            reg_max=reg_max,
            width_mult=wm,
        )

        self.postprocess = PostProcess(
            reg_max=reg_max,
            num_classes=num_classes,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )

        self.strides = [8, 16, 32]
        self._generate_anchors()

        self.cfg = cfg

    def _generate_anchors(self):
        """Precompute anchors for each feature level."""
        B = 1
        dummy = torch.zeros(B, 3, self.image_size, self.image_size)
        features = self._forward_backbone(dummy)
        feature_sizes = [(f.shape[2], f.shape[3]) for f in features]

        self.anchor_list = generate_anchors(
            feature_sizes, self.strides,
            self.image_size, device=next(self.parameters()).device
        )
        self.stride_tensor = torch.tensor(self.strides, dtype=torch.float32,
                                          device=next(self.parameters()).device)

    def _forward_backbone(self, x):
        return self.backbone(x)

    def _forward_neck(self, features):
        return self.neck(features)

    def _forward_head(self, features):
        return self.head(features)

    def forward(self, x, targets=None):
        """
        Forward pass.

        Args:
            x: Input tensor (B, 3, H, W)
            targets: Optional training targets (N, 6) [batch_idx, cls, cx, cy, w, h]
        Returns:
            If training: (o2m_preds, o2o_preds)
            If inference: list of dicts with 'boxes', 'scores', 'labels'
        """
        backbone_out = self._forward_backbone(x)
        neck_out = self._forward_neck(backbone_out)
        o2m_out, o2o_out = self._forward_head(neck_out)

        if self.training:
            return o2m_out, o2o_out

        results = self.postprocess(o2o_out, self.anchor_list,
                                   self.stride_tensor.tolist(),
                                   image_size=self.image_size, mode="o2o")
        return results

    def info(self, verbose=False):
        """Print model info."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cfg = self.cfg
        print(f"YOLOv26-{self.scale_name.upper()} | nc={self.nc} | scale={self.scale_name}")
        print(f"  Channels: {cfg['channels']}")
        print(f"  Depths: {cfg['depths']}")
        print(f"  Width mult: {cfg['width_mult']}")
        print(f"  Total params: {total:,}")
        print(f"  Trainable: {trainable:,}")
        return total

    def count_flops(self, input_size=640):
        """Estimate FLOPs for a forward pass."""
        h = w = input_size
        flops = 0
        for p in self.parameters():
            if p.grad is not None:
                continue
            flops += p.numel()
        return flops * h * w // (p.numel() if p.numel() > 0 else 1)


def build_yolo26(scale="n", num_classes=80, image_size=640, **kwargs):
    """Convenience factory function."""
    return YOLOv26Model(num_classes=num_classes, scale=scale,
                       image_size=image_size, **kwargs)
