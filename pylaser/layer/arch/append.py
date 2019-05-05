import torch

from ..base import Layer, summarize
from .sequence.sequence import Sequence


class Append(Layer):
    def __init__(self, *inner, axis=1):
        super().__init__()
        self.inner = Sequence(*inner)
        self.axis = axis

    def forward_inner(self, x):
        y = self.inner(x)
        return torch.cat([x, y], self.axis)

    def summarize_inner(self, num_percentiles=20):
        return {
            'inner': summarize(self.inner, num_percentiles),
            'axis': self.axis,
        }
