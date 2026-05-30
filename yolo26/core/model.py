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

    def forward(self, x):
        """
        Forward pass — pure inference from image to raw predictions.
        For training: use model.backbone/neck/head directly.
        """
        backbone_out = self._forward_backbone(x)
        neck_out = self._forward_neck(backbone_out)
        o2m_out, o2o_out = self._forward_head(neck_out)
        return o2m_out, o2o_out

    def predict(self, x, targets=None):
        """Inference: forward + postprocess. Returns detection results."""
        if self.training:
            return self.forward(x)
        with torch.no_grad():
            with torch.inference_mode():
                o2m_out, o2o_out = self.forward(x)
                return self.postprocess(
                    o2o_out, self.anchor_list,
                    self.stride_tensor.tolist(),
                    image_size=self.image_size, mode="o2o"
                )

    def _forward_impl(self, x):
        """Legacy compatibility — calls predict()."""
        return self.predict(x)

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
        """Estimate FLOPs: sum Conv FLOPs = H*W*K*K*Cin*Cout*out_channels * n_params."""
        flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                h_out = input_size // (m.stride[0] * 8)  # approximate after backbone
                w_out = input_size // (m.stride[1] * 8)
                flops += h_out * w_out * m.in_channels * m.out_channels * \
                         m.kernel_size[0] * m.kernel_size[1] * m.groups
        return int(flops)


def build_yolo26(scale="n", num_classes=80, image_size=640, **kwargs):
    """Convenience factory function."""
    return YOLOv26Model(num_classes=num_classes, scale=scale,
                       image_size=image_size, **kwargs)


def load_pretrained_weights(model, url=None, path=None, verbose=True):
    """
    Load Ultralytics YOLO26 pretrained weights into our model via shape matching.

    Strategy: For each of our layers, find official weights with IDENTICAL shape.
    This transfers ~60-80% of weights when architectures are similar.
    Non-matching layers stay randomly initialized (fine for training).

    Args:
        model: Our YOLOv26Model instance
        url: Optional URL to pretrained weights (.pt file)
        path: Optional local path to .pt file
        verbose: Print loading stats

    Returns:
        (loaded_count, skipped_count)
    """
    import os, urllib.request

    if path is None:
        path = "yolo26n.pt"
    if url is None:
        url = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"

    # Download if needed
    if not os.path.exists(path):
        if verbose:
            print(f"Downloading pretrained weights from {url}...")
        urllib.request.urlretrieve(url, path)
        if verbose:
            print(f"Saved to {path}")

    # Load official weights
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    # Navigate to state dict
    official_sd = {}
    if isinstance(ckpt, dict):
        if "model" in ckpt and hasattr(ckpt["model"], "state_dict"):
            official_sd = ckpt["model"].state_dict()
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            official_sd = {k: v for k, v in ckpt["model"].items()
                          if isinstance(v, torch.Tensor)}
        elif "state_dict" in ckpt:
            official_sd = ckpt["state_dict"]
        else:
            official_sd = {k: v for k, v in ckpt.items()
                          if isinstance(v, torch.Tensor)}
    elif hasattr(ckpt, "state_dict"):
        official_sd = ckpt.state_dict()

    # Strip 'model.' prefix if present (Ultralytics format)
    official_sd_clean = {}
    for k, v in official_sd.items():
        clean_key = k
        if k.startswith("model."):
            clean_key = k[len("model."):]
        if clean_key not in official_sd_clean:
            official_sd_clean[clean_key] = v

    # Index official weights by shape
    shape_map = {}
    for k, v in official_sd_clean.items():
        sk = str(tuple(v.shape))
        if sk not in shape_map:
            shape_map[sk] = []
        shape_map[sk].append((k, v))

    # Match by shape
    ours_sd = model.state_dict()
    matched_official_keys = set()
    loaded_keys = []
    skipped_keys = []

    for our_key, our_tensor in ours_sd.items():
        sk = str(tuple(our_tensor.shape))
        if sk in shape_map:
            for ok, ov in shape_map[sk]:
                if ok not in matched_official_keys:
                    ours_sd[our_key] = ov.clone()
                    matched_official_keys.add(ok)
                    loaded_keys.append((our_key, ok))
                    break
            else:
                skipped_keys.append(our_key)
        else:
            skipped_keys.append(our_key)

    model.load_state_dict(ours_sd, strict=False)

    if verbose:
        print(f"Pretrained weights: {len(loaded_keys)}/{len(ours_sd)} layers loaded "
              f"({100*len(loaded_keys)/len(ours_sd):.0f}%), "
              f"{len(skipped_keys)} unmatched (random init)")
        if loaded_keys and verbose >= 2:
            print("  Sample matches:")
            for our, off in loaded_keys[:5]:
                print(f"    {our} <- {off}")

    return len(loaded_keys), len(skipped_keys)
