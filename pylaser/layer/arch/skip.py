import torch
from torch import nn

from ...util.math import inverse_sigmoid
from ..base import Layer, summarize
from .iter.flow import Flow


class Skip(Layer):
    def __init__(self, *inner, usage=0.05):
        super().__init__()

        assert 0 < usage < 1
        x = torch.Tensor([usage])
        x = inverse_sigmoid(x)
        self.usage = nn.Parameter(x)

        self.inner = Flow(*inner)

    def forward_inner(self, x):
        usage = self.usage.sigmoid()
        return (1 - usage) * x + usage * self.inner(x)

    def summarize_inner(self, num_percentiles=20):
        raw_usage = self.usage
        return {
            'raw_usage': raw_usage.item(),
            'usage': raw_usage.sigmoid().item(),
            'inner': summarize(self.inner, num_percentiles),
        }
