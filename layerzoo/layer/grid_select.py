import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .base import Layer


class GridSelect(Layer):
    def __init__(self, in_shape, out_shape):
        super().__init__()
        if isinstance(in_shape, int):
            in_shape = in_shape,
        if isinstance(out_shape, int):
            out_shape = out_shape,
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.out_dim = np.prod(out_shape)
        self.sides = []
        inputs = []
        outputs = []
        for i, in_dim in enumerate(in_shape):
            side = nn.Parameter(torch.randn(in_dim, self.out_dim))
            self.sides.append(side)
            setattr(self, 'side_%d' % i, side)
            c = chr(ord('a') + i)
            inputs.append(c + 'z')
            outputs.append(c)
        inputs = ','.join(inputs)
        outputs = ''.join(outputs) + 'z'
        self.to_dense = '%s->%s' % (inputs, outputs)

    def forward_inner(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        weights = torch.einsum(self.to_dense, self.sides)
        weights = weights.view(-1, self.out_dim)
        weights = weights - weights.mean()
        weights = weights / (weights.std() + 1e-6)
        weights = weights ** 2
        x = torch.einsum('ni,io->no', [x, weights])
        out_shape = (batch_size,) + self.out_shape
        return x.view(*out_shape)
