import torch

from ..base import Layer, summarize
from ..reduce.simple.mean import ReduceMean


class WhileShuffled(Layer):
    def __init__(self, in_dim, out_dim=None, inner=None, reduce=None, axis=1):
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        if reduce is None:
            reduce = ReduceMean(axis)

        assert not out_dim % in_dim
        num_repeats = out_dim // in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_repeats = num_repeats
        self.axis = axis

        self.inner = inner
        self.reduce = reduce

        x = torch.randperm(num_repeats * in_dim) % in_dim
        self.register_buffer('indices', x)

        x = torch.zeros(in_dim * num_repeats, dtype=torch.int64)
        x[self.indices] = torch.arange(in_dim * num_repeats)
        self.register_buffer('inverse_indices', x)

    def forward_inner(self, x):
        assert x.shape[self.axis] == self.in_dim
        x = x.index_select(self.axis, self.indices)
        if self.inner:
            x = self.inner(x)
        x = x.index_select(self.axis, self.inverse_indices)
        shape = x.shape[:self.axis] + (self.num_repeats, self.in_dim) + \
            x.shape[self.axis + 1:]
        x = x.view(*shape)
        return self.reduce(x)

    def summarize_inner(self, num_percentiles=20):
        if self.inner:
            inner = summarize(self.inner, num_percentiles)
        else:
            inner = None
        return {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'num_repeats': self.num_repeats,
            'axis': self.axis,
            'inner': inner,
            'reduce': summarize(self.reduce, num_percentiles),
        }
