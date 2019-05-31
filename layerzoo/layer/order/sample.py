import torch
  
from ..base import Layer


class Sample(Layer):
    def __init__(self, in_dim, out_dim, axis=1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.axis = axis

        x = torch.randint(in_dim, (out_dim,))
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        assert x.shape[self.axis] == self.in_dim
        return x.index_select(self.axis, self.indices)
