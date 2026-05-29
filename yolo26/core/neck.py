import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import Conv, C3k2 as RefinedC3k2


class YOLOv26Neck(nn.Module):
    """
    YOLOv26 Neck: PAN-FPN (Path Aggregation + Feature Pyramid).
    Top-down FPN path + bottom-up PAN path.

    Args:
        in_chs: tuple of (p3_channels, p4_channels, p5_channels) from backbone
        out_ch: output channel width for all feature levels
    """

    def __init__(self, in_chs=(256, 512, 1024), out_ch=512):
        super().__init__()
        c3, c4, c5 = in_chs
        self.out_ch = out_ch

        # ── FPN Top-down path ──────────────────────────────────────────
        # P5 -> reduced P5
        self.reduce5 = Conv(c5, out_ch, 1, 1, act=True)
        # P4 + upsampled P5 -> fused P4
        self.c3k4 = RefinedC3k2(c4 + out_ch, out_ch, n=2, shortcut=False)
        # P3 + upsampled fused P4 -> fused P3
        self.c3k3 = RefinedC3k2(c3 + out_ch, out_ch, n=2, shortcut=False)

        # ── PAN Bottom-up path ─────────────────────────────────────────
        # fused P3 -> downsample -> fuse with fused P4
        self.downsample4 = Conv(out_ch, out_ch, 3, 2, act=True)
        self.c3k5 = RefinedC3k2(out_ch + out_ch, out_ch, n=2, shortcut=False)
        # fused P4 -> downsample -> fuse with P5
        self.downsample5 = Conv(out_ch, out_ch, 3, 2, act=True)
        self.c3k6 = RefinedC3k2(out_ch + c5, out_ch, n=2, shortcut=False)

    def forward(self, features):
        p3, p4, p5 = features

        # ── FPN: Top-down ───────────────────────────────────────────────
        fpn5 = self.reduce5(p5)                                    # c5 -> out_ch
        fpn4 = self.c3k4(torch.cat([
            F.interpolate(fpn5, size=p4.shape[-2:], mode='nearest'), p4], 1))
        fpn3 = self.c3k3(torch.cat([
            F.interpolate(fpn4, size=p3.shape[-2:], mode='nearest'), p3], 1))

        # ── PAN: Bottom-up ─────────────────────────────────────────────
        pan4 = self.c3k5(torch.cat([self.downsample4(fpn3), fpn4], 1))
        pan5 = self.c3k6(torch.cat([self.downsample5(pan4), p5], 1))

        return fpn3, pan4, pan5
