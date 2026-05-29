import torch
import torch.nn as nn
from .blocks import Conv, C3k2 as RefinedC3k2


class YOLOv26Neck(nn.Module):
    """
    YOLOv26 Neck: PAN-FPN (Path Aggregation + Feature Pyramid).
    Top-down FPN path + bottom-up PAN path.

    Args:
        in_chs: tuple of (p3_channels, p4_channels, p5_channels) from backbone
        out_ch: output channel width for all feature levels
    """

    def __init__(self, in_chs=(64, 128, 256), out_ch=64, width_mult=1.0):
        super().__init__()
        c3, c4, c5 = in_chs
        self.out_ch = out_ch

        # ── FPN Top-down path ──────────────────────────────────────────
        # P5 -> reduced P5
        self.reduce5 = Conv(c5, out_ch, 1, 1, act=True)
        # P4 + upsampled P5 -> fused P4
        self.upsample4 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k4 = RefinedC3k2(c4 + out_ch, out_ch, n=2, shortcut=False, width_mult=1.0, act=True)
        # P3 + upsampled fused P4 -> fused P3
        self.upsample3 = nn.Upsample(scale_factor=2, mode="nearest")
        self.c3k3 = RefinedC3k2(c3 + out_ch, out_ch, n=2, shortcut=False, width_mult=1.0, act=True)

        # ── PAN Bottom-up path ─────────────────────────────────────────
        # fused P3 -> downsample -> fuse with fused P4
        self.downsample4 = Conv(out_ch, out_ch, 3, 2, act=True)
        self.c3k5 = RefinedC3k2(out_ch + out_ch, out_ch, n=2, shortcut=False, width_mult=1.0, act=True)
        # fused P4 -> downsample -> fuse with P5
        self.downsample5 = Conv(out_ch, out_ch, 3, 2, act=True)
        self.c3k6 = RefinedC3k2(out_ch + c5, out_ch, n=2, shortcut=False, width_mult=1.0, act=True)

    def forward(self, features):
        p3, p4, p5 = features

        # ── FPN: Top-down ───────────────────────────────────────────────
        fpn5 = self.reduce5(p5)                          # c5 -> out_ch
        fpn4 = self.c3k4(torch.cat([self.upsample4(fpn5), p4], 1))   # c4+out_ch -> out_ch
        fpn3 = self.c3k3(torch.cat([self.upsample3(fpn4), p3], 1))   # c3+out_ch -> out_ch

        # ── PAN: Bottom-up ────────────────────────────────────────────
        pan4 = self.c3k5(torch.cat([self.downsample4(fpn3), fpn4], 1))  # out_ch+out_ch -> out_ch
        pan5 = self.c3k6(torch.cat([self.downsample5(pan4), p5], 1))    # out_ch+c5 -> out_ch

        return fpn3, pan4, pan5
