import torch
  
from ...base import Layer, summarize
from ...order.sample import Sample
from .flow import Flow


class PercolateStage(Layer):
    def __init__(self, inner, dim, inner_out_dim, stage_id):
        super().__init__()
        rate = inner_out_dim / dim
        sampled_inner_out_dim = self.compute_num_overwrites(dim, rate, stage_id)
        self.in_dim = dim
        self.inner_in_dim = dim
        self.inner_out_dim = inner_out_dim
        self.sampled_inner_out_dim = sampled_inner_out_dim
        self.out_dim = dim
        self.stage_id = stage_id
        self.rate = rate
        self.inner = inner
        self.expand_inner_output = Sample(inner_out_dim, sampled_inner_out_dim)
        self.input_to_keep = Sample(dim, dim - sampled_inner_out_dim)

    @classmethod
    def compute_num_overwrites(cls, size, rate, stage_id):
        cross_over = int(1 / rate)
        if stage_id < cross_over:
            count = int(size / (stage_id + 1))
        else:
            count = int(size * rate)
        return count

    def forward_inner(self, x):
        new = self.inner(x)
        expanded_new = self.expand_inner_output(new)
        old = self.input_to_keep(x)
        return torch.cat([old, expanded_new], 1)

    def summarize_inner(self, num_percentiles=20):
        inner = summarize(self.inner, num_percentiles)
        expand_inner_output = summarize(self.expand_inner_output,
                                        num_percentiles)
        input_to_keep = summarize(self.input_to_keep, num_percentiles)
        return {
            'in_dim': self.in_dim,
            'inner_in_dim': self.inner_in_dim,
            'inner_out_dim': self.inner_out_dim,
            'sampled_inner_out_dim': self.sampled_inner_out_dim,
            'out_dim': self.out_dim,
            'stage_id': self.stage_id,
            'rate': self.rate,
            'inner': inner,
            'expand_inner_output': expand_inner_output,
            'input_to_keep': input_to_keep,
        }


class Percolate(Flow):
    def __init__(self, num_stages, new_inner, dim, inner_out_dim):
        stages = []
        for i in range(num_stages):
            inner = new_inner(dim, inner_out_dim)
            stage = PercolateStage(inner, dim, inner_out_dim, i)
            stages.append(stage)
        super().__init__(*stages)
        self.num_stages = num_stages
        self.new_inner = new_inner
        self.in_dim = dim
        self.inner_in_dim = dim
        self.inner_out_dim = inner_out_dim
        self.out_dim = dim

    def summarize_inner(self, num_percentiles=20):
        stages = self.summarize_layers(num_percentiles)
        return {
            'stages': stages,
            'num_stages': self.num_stages,
            'in_dim': self.in_dim,
            'inner_in_dim': self.inner_in_dim,
            'inner_out_dim': self.inner_out_dim,
            'out_dim': self.out_dim,
        }
