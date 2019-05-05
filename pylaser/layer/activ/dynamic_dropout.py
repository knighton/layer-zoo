import torch
from torch import nn

from ..base import  Layer
from ...util.math import inverse_sigmoid


class DynamicDropout(Layer):
    def __init__(self, compute_rate, rate_mul=0.05, sharpness=10):
        super().__init__()
        self.compute_rate = compute_rate
        x = torch.Tensor([rate_mul])
        x = inverse_sigmoid(x)
        self.raw_rate_mul = nn.Parameter(x)
        x = torch.Tensor([sharpness])
        x = x.log()
        self.raw_sharpness = nn.Parameter(x)

    def forward_inner(self, x):
        rate_mul = self.raw_rate_mul.sigmoid()
        base_rate = self.compute_rate(x)
        rate = base_rate * rate_mul
        keep_prob = 1 - rate
        if self.training:
            noise = torch.rand(*x.shape)
            if x.is_cuda:
                noise = noise.cuda()
            keptness = keep_prob - noise
        else:
            keptness = keep_prob - 0.5
        sharpness = self.raw_sharpness.exp()
        is_kept = (keptness * sharpness).sigmoid()
        return x * is_kept

    def summarize_inner(self, num_percentiles=20):
        return {
            'raw_rate_mul': self.raw_rate_mul.item(),
            'rate_mul': self.raw_rate_mul.detach().sigmoid().item(),
        }
