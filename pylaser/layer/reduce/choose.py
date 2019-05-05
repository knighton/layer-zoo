import torch
from torch import nn
from torch.nn import functional as F

from .base import Reduce


class ReduceChoose(Reduce):
    def __init__(self, axis, dim, keepdim=False):
        super().__init__(axis, keepdim)
        x = torch.zeros(axis)
        self.raw_usage = nn.Parameter(x)

    def forward_inner(self, x):
        usage = F.softmax(self.raw_usage, 0)
        pre_shape = x.shape[:self.axis]
        post_shape = x.shape[self.axis:]
        shape = pre_shape + (self.dim,) + post_shape
        usage = usage.view(*shape)
        return (usage * x).sum(self.axis)

    def summarize_inner(self, num_percentiles=20):
        raw_usage = self.raw_usage.detach().cpu().numpy().tolist()
        usage = F.softmax(self.raw_usage.detach(), 0)
        usage = usage.cpu().numpy().tolist()
        return {
            'raw_uage': raw_usage,
            'usage': usage,
        }
