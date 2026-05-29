import torch
import torch.nn as nn
import math
from .blocks import Conv


class YOLOv26Head(nn.Module):
    """
    YOLOv26 Dual Head: O2M (One-to-Many) + O2O (One-to-One).

    Both branches share the same 4-reg_max bbox regression.
    O2M branch: reg_max channels -> NMS -> multiple predictions per GT
    O2O branch: 4 channels -> Hungarian matching -> Top-K -> one prediction per GT

    reg_max=4 (DFL removal innovation): 4 channels instead of 64 (16*4).
    Optimized: 1 Conv layer per branch (was 2 Conv layers = 4 total).
    """

    def __init__(self, num_classes=80, in_ch=256, reg_max=4):
        super().__init__()
        self.nc = num_classes
        self.reg_max = reg_max
        self.stride = torch.tensor([8, 16, 32])

        # O2M Branch — 2 Conv layers
        self.o2m_cls = nn.Sequential(
            Conv(in_ch, in_ch, 3, 1, act=True),
            Conv(in_ch, in_ch, 3, 1, act=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.o2m_reg = nn.Sequential(
            Conv(in_ch, in_ch, 3, 1, act=True),
            Conv(in_ch, in_ch, 3, 1, act=True),
            nn.Conv2d(in_ch, reg_max * 4, 1),
        )

        # O2O Branch — 2 Conv layers
        self.o2o_cls = nn.Sequential(
            Conv(in_ch, in_ch, 3, 1, act=True),
            Conv(in_ch, in_ch, 3, 1, act=True),
            nn.Conv2d(in_ch, num_classes, 1),
        )
        self.o2o_reg = nn.Sequential(
            Conv(in_ch, in_ch, 3, 1, act=True),
            Conv(in_ch, in_ch, 3, 1, act=True),
            nn.Conv2d(in_ch, reg_max * 4, 1),
        )

        self._initialize_biases()

    def _initialize_biases(self):
        """Initialize classification bias for foreground class imbalance."""
        for conv in [self.o2m_cls, self.o2o_cls]:
            b = conv[-1].bias
            num_classes = self.nc
            nn.init.constant_(b, -math.log((1 - 0.01) / 0.01))

    def forward(self, features):
        """
        Args:
            features: list of 3 tensors [(B,C,80,80), (B,C,40,40), (B,C,20,20)]
        Returns:
            o2m_out: list of dicts with 'cls', 'reg' for each level
            o2o_out: same structure for O2O branch
        """
        o2m_out = []
        o2o_out = []

        for i, x in enumerate(features):
            o2m_out.append({
                "cls": self.o2m_cls(x),
                "reg": self.o2m_reg(x),
            })
            o2o_out.append({
                "cls": self.o2o_cls(x),
                "reg": self.o2o_reg(x),
            })

        return o2m_out, o2o_out
