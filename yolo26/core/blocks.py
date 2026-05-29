import torch
import torch.nn as nn


class Conv(nn.Module):
    """Standard convolution: Conv2d + BatchNorm + SiLU."""

    def __init__(self, in_ch, out_ch, kernel_size=1, stride=1, padding=None,
                 groups=1, dilation=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride,
                              padding, dilation=dilation, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Residual bottleneck: 1x1 conv -> 3x3 conv -> 1x1 conv with skip connection."""

    def __init__(self, in_ch, out_ch, shortcut=True, expansion=0.5, act=True):
        super().__init__()
        hidden_ch = max(int(out_ch * expansion), 1)
        self.cv1 = Conv(in_ch, hidden_ch, 1, 1, act=act)
        self.cv2 = Conv(hidden_ch, out_ch, 3, 1, act=act)
        self.add = shortcut and in_ch == out_ch

    def forward(self, x):
        h = self.cv2(self.cv1(x))
        return x + h if self.add else h


class C3k2(nn.Module):
    """
    C3k2: CSP Bottleneck stack with 2 convolutions (YOLOv8 C2f-style).

    Design (proven, simple):
        - cv1: in_ch -> (n+1)*c
        - n bottleneck blocks, each taking c channels and outputting c channels,
          with the output of each feeding into the next bottleneck
        - b: a copy of the input for passthrough
        - Concatenate [b, m0, m1, ..., m_{n-1}] -> (n+1)*c
        - cv2: (n+1)*c -> out_ch

    Where c = out_ch * width_mult.

    Args:
        in_ch: input channels
        out_ch: output channels
        n: number of bottleneck layers in the CSP stack
        shortcut: residual connections in bottleneck
        width_mult: channel width multiplier
        act: activation
    """

    def __init__(self, in_ch, out_ch, n=1, shortcut=False, width_mult=1.0, act=True):
        super().__init__()
        c = int(out_ch * width_mult)
        self.n = n
        self.c = c
        # Project: in_ch -> (n+1)*c
        self.cv1 = Conv(in_ch, (n + 1) * c, 1, 1, act=act)
        # Project: (n+1)*c -> out_ch
        self.cv2 = Conv((n + 1) * c, out_ch, 1, act=act)
        # n bottlenecks, each: c -> c (expansion=1.0 keeps channel count)
        self.m = nn.ModuleList(
            Bottleneck(c, c, shortcut=shortcut, expansion=1.0, act=act)
            for _ in range(n)
        )

    def forward(self, x):
        # cv1: in_ch -> (n+1)*c
        y = self.cv1(x)          # (B, (n+1)*c, H, W)
        # Split y into n chunks of c channels each: [chunk_0, ..., chunk_{n-1}]
        chunks = list(y.split(self.c, dim=1))  # list of n tensors, each (B, c, H, W)
        # Passthrough copy of input for the "b" branch
        outs = [chunks[0].clone()]  # b = passthrough of first chunk
        # Feed each subsequent chunk through its bottleneck
        for i in range(self.n):
            outs.append(self.m[i](chunks[i]))
        # Concatenate: [b, m0(chunk_0), m1(chunk_1), ...] -> (n+1)*c
        return self.cv2(torch.cat(outs, dim=1))


# Alias
RefinedC3k2 = C3k2


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (3x maxpool 5x5)."""

    def __init__(self, in_ch, out_ch, kernel_size=5, act=True):
        super().__init__()
        hidden_ch = max(in_ch // 2, 1)
        self.cv1 = Conv(in_ch, hidden_ch, 1, 1, act=act)
        self.cv2 = Conv(hidden_ch * 4, out_ch, 1, 1, act=act)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        y3 = self.m(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class DWConv(Conv):
    """Depthwise convolution (groups=in_ch)."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, act=True):
        super().__init__(in_ch, out_ch, kernel_size, stride, groups=in_ch, act=act)
