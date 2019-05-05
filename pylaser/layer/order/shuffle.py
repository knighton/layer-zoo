import torch
  
from ..base import Layer


class Shuffle(Layer):
    def __init__(self, in_dim, out_dim=None, axis=1):
        super().__init__()

        if out_dim is None:
            out_dim = in_dim

        assert not out_dim % in_dim
        num_repeats = out_dim // in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_repeats = num_repeats
        self.axis = axis

        x = torch.randperm(num_repeats * in_dim) % in_dim
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        assert x.shape[self.axis] == self.in_dim
        return x.index_select(self.axis, self.indices)

    def json_inner(self):
        return {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'num_repeats': self.num_repeats,
            'axis': self.axis,
        }
