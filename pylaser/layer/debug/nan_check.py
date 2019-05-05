import torch

from ..base import Layer


class NaNCheck(Layer):
    def forward_inner(self, x):
        assert not torch.isnan(x).any()
        return x
