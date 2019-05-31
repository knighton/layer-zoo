import torch
from torch import nn

from .base import Layer


class InRange(Layer):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape
        self.first_limit = nn.Parameter(torch.randn(*shape))
        self.first_polarity = nn.Parameter(torch.randn(*shape))
        self.second_limit = nn.Parameter(torch.randn(*shape))
        self.second_polarity = nn.Parameter(torch.randn(*shape))

    def forward_inner(self, x):
        first = (x - self.first_limit).tanh() * self.first_polarity
        second = (x - self.second_limit).tanh() * self.second_polarity
        return first * second
