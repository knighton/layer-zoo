import torch
from torch import nn
  
from ...base import Layer, summarize
from .flow import Flow


class AccumulateStage(Layer):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward_inner(self, x):
        y = self.inner(x)
        return torch.cat([x, y], 1)

    def summarize_inner(self, num_percentiles):
        return summarize(self.inner, num_percentiles)


class Accumulate(Flow):
    def __init__(self, num_stages, new_inner, in_dim, inner_out_dim):
        stages = []
        for i in range(num_stages):
            inner_in_dim = in_dim + i * inner_out_dim
            inner = new_inner(inner_in_dim, inner_out_dim)
            stage = AccumulateStage(inner)
            stages.append(stage)
        super().__init__(*stages)
        self.num_stages = num_stages
        self.new_inner = new_inner
        self.in_dim = in_dim
        self.inner_out_dim = inner_out_dim
        self.out_dim = in_dim + num_stages * inner_out_dim

    def summarize_inner(self, num_percentiles=20):
        stages = self.summarize_layers(num_percentiles)
        return {
            'stages': stages,
            'num_stages': self.num_stages,
            'in_dim': self.in_dim,
            'inner_out_dim': self.inner_out_dim,
            'out_dim': self.out_dim,
        }
