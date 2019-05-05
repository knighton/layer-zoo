import torch
from torch import nn

from .base import Layer


class Adjust(Layer):
    def __init__(self, *shape):
        super().__init__()

        x = torch.randn(*shape) * 0.05
        self.pre_add = nn.Parameter(x)

        x = torch.randn(*shape) * 0.05 + 1
        self.mul = nn.Parameter(x)

        x = torch.randn(*shape) * 0.05
        self.post_add = nn.Parameter(x)

    def forward_inner(self, x):
        return (x + self.pre_add) * self.mul + self.post_add
