import torch
from torch import nn

from .base import Layer


class Hinge(Layer):
    def __init__(self, neg=0, pos=1):
        super().__init__()
        self.neg = nn.Parameter(torch.Tensor([neg]))
        self.pos = nn.Parameter(torch.Tensor([pos]))

    def forward_inner(self, x):
        neg = (x < 0).float() * self.neg * x
        pos = (0 < x).float() * self.pos * x
        return neg + pos

    def summarize(self, num_percentiles=20):
        return {
            'neg': self.neg.item(),
            'pos': self.pos.item(),
        }
