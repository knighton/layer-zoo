import torch

from ..base import Layer, summarize
from .iter.flow import Flow


class Append(Layer):
    def __init__(self, *inner, axis=1):
        super().__init__()
        self.inner = Flow(*inner)
        self.axis = axis

    def forward_inner(self, x):
        y = self.inner(x)
        return torch.cat([x, y], self.axis)

    def summarize_inner(self, num_percentiles=20):
        return {
            'inner': summarize(self.inner, num_percentiles),
            'axis': self.axis,
        }
