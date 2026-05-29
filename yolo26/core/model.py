import torch
import torch.nn as nn
import math
from .backbone import YOLOv26Backbone
from .neck import YOLOv26Neck
from .head import YOLOv26Head
from .postprocess import PostProcess


SCALE_CONFIGS = {
    # YOLOv26 scales — optimized for Apple MPS (M4)
    # X scale removed (1.3B params not practical for edge/mobile)
    # channels are already ACTUAL values (width_mult pre-applied)
    "n": {
        "channels": (16, 32, 64, 128, 256),   # 32-aligned, ~4.5M params
        "depths": (1, 1, 3, 3, 1),          # -1 layer per P3/P4 bottleneck
        "width_mult": 0.25,
        "num_repeats": (1, 2, 4, 4, 2),
    },
    "s": {
        "channels": (32, 64, 128, 256, 512),   # 32-aligned, ~30M params
        "depths": (1, 3, 6, 6, 3),           # original
        "width_mult": 0.5,
        "num_repeats": (1, 3, 9, 9, 3),
    },
    "m": {
        "channels": (60, 120, 240, 480, 960),  # original COCO scaling, ~133M params
        "depths": (2, 4, 8, 8, 4),           # original
        "width_mult": 0.75,
        "num_repeats": (2, 6, 12, 12, 4),
    },
    "l": {
        "channels": (96, 192, 384, 768, 1536), # original, ~479M params
        "depths": (3, 6, 12, 12, 6),          # original
        "width_mult": 1.0,
        "num_repeats": (3, 9, 18, 18, 6),
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

        # SCALE_CONFIGS stores ACTUAL channel counts (post-width_mult).
        # Backbone receives them directly — no additional multiplication.
        self.backbone = YOLOv26Backbone(
            in_ch=3,
            channels=cfg["channels"],
            depths=cfg["depths"],
            width_mult=1.0,
        )

        # Neck in_chs from backbone output (P3, P4, P5).
        # neck out_ch follows original formula: int(256 * wm)
        wm = cfg["width_mult"]
        neck_out_ch = int(256 * wm)

        self.neck = YOLOv26Neck(
            in_chs=cfg["channels"][2:],  # (c3, c4, c5) from backbone
            out_ch=neck_out_ch,
        )

        self.head = YOLOv26Head(
            num_classes=num_classes,
            in_ch=neck_out_ch,
            reg_max=reg_max,
        )

        self.postprocess = PostProcess(
            reg_max=reg_max,
            num_classes=num_classes,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )

        self._apply_hardcoded_optimizations()

        self.strides = [8, 16, 32]
        # Hardcoded anchor values: no dummy forward pass needed.
        # image_size=640, strides=[8,16,32], features=[80x80, 40x40, 20x20]
        self._generate_anchors()

        self.cfg = cfg

    def _generate_anchors(self):
        """Precompute anchors for each feature level.

        Hardcoded for image_size=640, strides=[8,16,32]:
          P3: 80x80 grid, stride 8  → 6400 anchors
          P4: 40x40 grid, stride 16 → 1600 anchors
          P5: 20x20 grid, stride 32 → 400 anchors
        Each anchor box = [cx_norm, cy_norm, w_norm, h_norm] in [0,1].
        """
        img_size = self.image_size
        device = next(self.parameters()).device
        self.anchor_list = []
        self.stride_tensor = torch.tensor(self.strides, dtype=torch.float32, device=device)

        for stride in self.strides:
            grid_size = img_size // stride
            anchor_size = stride / img_size

            shifts_x = (torch.arange(grid_size, device=device) + 0.5) * stride / img_size
            shifts_y = (torch.arange(grid_size, device=device) + 0.5) * stride / img_size
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            anchor = torch.stack([
                shift_x,
                shift_y,
                torch.full_like(shift_x, anchor_size),
                torch.full_like(shift_x, anchor_size),
            ], dim=1)
            self.anchor_list.append(anchor)

    def _apply_hardcoded_optimizations(self):
        """Apply MPS hardware optimizations after model construction.

        torch.compile is disabled on MPS — MPS doesn't benefit from it
        and can cause significant overhead on larger scales.
        """
        pass

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
        if not self.training:
            with torch.no_grad():
                with torch.inference_mode():
                    return self._forward_impl(x)
        return self._forward_impl(x)

    def _forward_impl(self, x):
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
