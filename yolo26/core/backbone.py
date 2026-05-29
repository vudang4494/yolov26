import torch
import torch.nn as nn
from .blocks import Conv, RefinedC3k2, SPPF


class YOLOv26Backbone(nn.Module):
    """
    YOLOv26 Backbone: CSPDarknet with RefinedC3k2 blocks.
    Outputs P3(80x80), P4(40x40), P5(20x20) for input 640x640.

    Spatial flow for 640x640 input:
        Stem:  3 -> c1 (stride 2) -> 320x320
        Stage1: c1 -> c2 (stride 2) -> 160x160
        Stage2: c2 -> c3 (stride 2) -> 80x80   (P3)
        Stage3: c3 -> c4 (stride 2) -> 40x40   (P4)
        Stage4: c4 -> c5 (stride 2) -> 20x20   (P5)
    """

    def __init__(self, in_ch=3, channels=(64, 128, 256, 512, 1024),
                 depths=(1, 3, 6, 6, 3), width_mult=1.0):
        super().__init__()
        c1, c2, c3, c4, c5 = [int(c * width_mult) for c in channels]

        self.stem = Conv(in_ch, c1, 3, 2, act=True)

        self.stage1 = nn.Sequential(
            Conv(c1, c2, 3, 2, act=True),
            RefinedC3k2(c2, c2, n=depths[0], shortcut=True, width_mult=1.0, act=True),
        )

        self.stage2 = nn.Sequential(
            Conv(c2, c3, 3, 2, act=True),
            RefinedC3k2(c3, c3, n=depths[1], shortcut=True, width_mult=1.0, act=True),
        )

        self.stage3 = nn.Sequential(
            Conv(c3, c4, 3, 2, act=True),
            RefinedC3k2(c4, c4, n=depths[2], shortcut=True, width_mult=1.0, act=True),
        )

        self.stage4 = nn.Sequential(
            Conv(c4, c5, 3, 2, act=True),
            RefinedC3k2(c5, c5, n=depths[3], shortcut=True, width_mult=1.0, act=True),
            SPPF(c5, c5, kernel_size=5, act=True),
        )

        self.out_chs = (c3, c4, c5)

    def forward(self, x):
        x = self.stem(x)      # 320x320
        p2 = self.stage1(x)   # 160x160
        p3 = self.stage2(p2)  # 80x80  -> P3
        p4 = self.stage3(p3)  # 40x40  -> P4
        p5 = self.stage4(p4)  # 20x20  -> P5
        return p3, p4, p5
