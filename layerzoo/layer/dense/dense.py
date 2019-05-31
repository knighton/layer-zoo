import torch
from torch import nn

from ..base import Layer


class Dense(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        x = torch.randn(in_dim) * 0.05
        self.pre_bias = nn.Parameter(x)

        x = torch.randn(in_dim, out_dim) * 0.05
        self.kernel = nn.Parameter(x)

        x = torch.randn(out_dim) * 0.05
        self.post_bias = nn.Parameter(x)

    def forward_inner(self, x):
        return torch.einsum('ni,i,io,o->no',
                            [x, self.pre_bias, self.kernel, self.post_bias])

    def summarize_inner(self, num_percentiles=20):
        return {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
        }
