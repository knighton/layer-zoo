import torch
from torch import nn

from ..base import Layer


class CuilDense(Layer):
    def __init__(self, cuils, in_dim, out_dim):
        super().__init__()
        self.cuils = cuils
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.pre_add = nn.Parameter(torch.randn(in_dim))
        self.mul = nn.Parameter(torch.randn(cuils, in_dim, out_dim))
        self.post_add = nn.Parameter(torch.randn(out_dim))

    def forward_inner(self, x):
        x = torch.einsum('nei,i,eio,o->neo',
                         [x, self.pre_add, self.mul, self.post_add])
        return x.contiguous()
