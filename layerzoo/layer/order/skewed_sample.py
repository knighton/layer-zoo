import numpy as np
import torch

from ..base import Layer


def make_indices(in_dim, out_dim, skew, integerization_leeway=100):
    # Get a pointy distribution.
    x = np.random.normal(0, 1, in_dim)
    x = np.abs(x)

    # Turn that into number of slots assigned to each index.
    x **= skew
    x /= x.sum()
    x *= out_dim * integerization_leeway

    # Show how counts are distributed for debug.
    # a, b = np.histogram(x.astype(np.int64), 30)
    # print(a)
    # print(b)

    # Create the skewed pool of indices.
    indices = []
    for index, value in enumerate(x):
        indices += [index] * int(value)
    indices = np.array(indices, np.int64)

    # Sample out_dim values from it.
    return np.random.choice(indices, out_dim)


class SkewedSample(Layer):
    def __init__(self, in_dim, out_dim, skew, axis=1):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.skew = skew
        self.axis = axis
        x = make_indices(in_dim, out_dim, skew)
        x = torch.from_numpy(x)
        self.register_buffer('indices', x)

    def forward_inner(self, x):
        assert x.shape[self.axis] == self.in_dim
        return x.index_select(self.axis, self.indices)
