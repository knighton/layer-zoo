import torch

from ..util.math import int_randn
from .base import Layer


def make_indices1d(in_channels, out_channels, width, width_std):
    out_shape = out_channels, width

    c = torch.randint(in_channels, out_shape)
    c = c * width

    w_center = torch.arange(width).view(1, 1, -1)
    w_offset = int_randn(out_shape, width_std)
    w = w_center + w_offset
    w = w.clamp(0, width - 1)

    indices = c + w
    return indices.view(-1)


def make_indices2d(in_channels, out_channels, height, width, height_std,
                   width_std):
    out_shape = out_channels, height, width

    c = torch.randint(in_channels, out_shape)
    c = c * height * width

    h_center = torch.arange(height).view(1, -1, 1)
    h_offset = int_randn(out_shape, height_std)
    h = h_center + h_offset
    h = h.clamp(0, height - 1)
    h = h * width

    w_center = torch.arange(width).view(1, 1, -1)
    w_offset = int_randn(out_shape, width_std)
    w = w_center + w_offset
    w = w.clamp(0, width - 1)

    indices = c + h + w
    return indices.view(-1)


def make_indices3d(in_channels, out_channels, depth, height, width, depth_std,
                   height_std, width_std):
    out_shape = out_channels, depth, height, width

    c = torch.randint(in_channels, out_shape)
    c = c * depth * height * width

    d_center = torch.arange(depth).view(1, -1, 1)
    d_offset = int_randn(out_shape, depth_std)
    d = d_center + d_offset
    d = d.clamp(0, depth - 1)
    d = d * height * width

    h_center = torch.arange(height).view(1, -1, 1)
    h_offset = int_randn(out_shape, height_std)
    h = h_center + h_offset
    h = h.clamp(0, height - 1)
    h = h * width

    w_center = torch.arange(width).view(1, 1, -1)
    w_offset = int_randn(out_shape, width_std)
    w = w_center + w_offset
    w = w.clamp(0, width - 1)

    indices = c + d + h + w
    return indices.view(-1)


class Scatter1d(Layer):
    def __init__(self, in_channels, out_channels, width, width_std=None):
        super().__init__()

        if width_std is None:
            width_std = int(width ** 0.5)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.width_std = width_std

        x = make_indices1d(in_channels, out_channels, width, width_std)
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        n, c, w = x.shape
        x = x.view(n, -1)
        y = x.index_select(1, self.indices)
        return y.view(n, -1, w)


class Scatter2d(Layer):
    def __init__(self, in_channels, out_channels, height, width,
                 height_std=None, width_std=None):
        super().__init__()

        if height_std is None:
            height_std = int(height ** 0.5)
        if width_std is None:
            width_std = int(width ** 0.5)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.height = height
        self.width = width
        self.height_std = height_std
        self.width_std = width_std

        x = make_indices2d(in_channels, out_channels, height, width, height_std,
                           width_std)
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        n, c, h, w = x.shape
        x = x.view(n, -1)
        y = x.index_select(1, self.indices)
        return y.view(n, -1, h, w)


class Scatter3d(Layer):
    def __init__(self, in_channels, out_channels, depth, height, width,
                 depth_std=None, height_std=None, width_std=None):
        super().__init__()

        if depth_std is None:
            depth_std = int(depth ** 0.5)
        if height_std is None:
            height_std = int(height ** 0.5)
        if width_std is None:
            width_std = int(width ** 0.5)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.height = height
        self.width = width
        self.depth_std = depth_std
        self.height_std = height_std
        self.width_std = width_std

        x = make_indices3d(in_channels, out_channels, depth, height, width,
                           depth_std, height_std, width_std)
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        n, c, d, h, w = x.shape
        x = x.view(n, -1)
        y = x.index_select(1, self.indices)
        return y.view(n, -1, d, h, w)
