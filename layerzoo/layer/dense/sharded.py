import torch
from torch import nn

from ..base import Layer


class ShardedDense(Layer):
    def __init__(self, in_dim, out_dim, shard_in_dim):
        super().__init__()

        assert not in_dim % shard_in_dim
        num_shards = in_dim // shard_in_dim
        assert not out_dim % num_shards
        shard_out_dim = out_dim // num_shards

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_shards = num_shards
        self.shard_in_dim = shard_in_dim
        self.shard_out_dim = shard_out_dim

        x = torch.randn(shard_in_dim, num_shards)
        self.pre_bias = nn.Parameter(x)

        x = torch.randn(shard_in_dim, shard_out_dim, num_shards)
        self.kernel = nn.Parameter(x)

        x = torch.randn(shard_out_dim, num_shards)
        self.post_bias = nn.Parameter(x)

    def forward_inner(self, x):
        batch_size, dim = x.shape
        x = x.view(batch_size, self.shard_in_dim, self.num_shards)
        x = torch.einsum('nis,is,ios,os->nos',
                         [x, self.pre_bias, self.kernel, self.post_bias])
        return x.view(batch_size, self.out_dim)

    def summarize_inner(self, num_percentiles=20):
        return {
            'in_dim': self.in_dim,
            'out_dim': self.out_dim,
            'num_shards': self.num_shards,
            'shard_in_dim': self.shard_in_dim,
            'shard_out_dim': self.shard_out_dim,
        }
