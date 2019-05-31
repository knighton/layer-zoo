import torch
from torch import nn
from torch.nn import functional as F

from ..util.layer import normalize_coords
from .base import Layer


class Blur2d(Layer):
    def __init__(self, channels, height, width, backoff=1.5, kernel_size=7):
        super().__init__()
        kernel_height, kernel_width = kernel_size = \
            normalize_coords(kernel_size, 2)
        assert 1 <= kernel_height
        assert kernel_height % 2
        assert 1 <= kernel_width
        assert kernel_width % 2
        self.channels = channels
        self.height = height
        self.width = width

        assert 0 < backoff
        x = torch.Tensor([backoff])
        x = x.log()
        self.raw_backoff = nn.Parameter(x)

        self.kernel_size = kernel_size

        self.padding = kernel_height // 2, kernel_width // 2

        heights = torch.arange(kernel_height).float() - kernel_height // 2
        heights = heights.unsqueeze(1)
        widths = torch.arange(kernel_width).float() - kernel_width // 2
        widths = widths.unsqueeze(0)
        distances = (heights ** 2 + widths ** 2) ** 0.5
        base_kernel = distances.view(1, 1, kernel_height, kernel_width)
        self.register_buffer('base_kernel', base_kernel)

    def forward_inner(self, x):
        backoff = self.raw_backoff.exp()
        kernel = self.base_kernel * backoff
        kernel = 1 - kernel.tanh()
        kernel = kernel / kernel.sum()
        batch_size, channels, height, width = x.shape
        x = x.view(-1, 1, height, width)
        x = F.conv2d(x, kernel, padding=self.padding)
        return x.view(batch_size, channels, height, width)

    def summarize_inner(self, num_percentiles=20):
        return {
            'channels': self.channels,
            'height': self.height,
            'width': self.width,
            'backoff': self.raw_backoff.exp().item(),
            'kernel_size': self.kernel_size,
            'padding': self.padding,
        }
