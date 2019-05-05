import torch
from torch import nn
from torch.nn import functional as F

from .base import Reduce


class ReduceLottery(Reduce):
    def __init__(self, axis, dim, keepdim=False, sharpness=10):
        super().__init__(axis, keepdim)
        self.dim = dim
        x = torch.zeros(dim)
        self.raw_usage = nn.Parameter(x)
        x = torch.Tensor([sharpness])
        x = x.log()
        self.raw_sharpness = nn.Parameter(x)

    def forward_inner(self, x):
        usage = F.softmax(self.raw_usage, 0)
        pre_shape = (1,) * self.axis
        post_shape = (1,) * (len(x.shape) - self.axis - 1)
        shape = pre_shape + (self.dim,) + post_shape
        keep_prob = usage.view(*shape)
        if self.training:
            noise = torch.rand(*x.shape)
            if x.is_cuda:
                noise = noise.cuda()
            keptness = keep_prob - noise
        else:
            keptness = keep_prob - 0.5
        sharpness = self.raw_sharpness.exp()
        is_kept = (keptness * sharpness).sigmoid()
        return (x * is_kept).sum(self.axis)

    def summarize_inner(self, num_percentiles=20):
        raw_usage = self.raw_usage.detach().cpu().numpy().tolist()
        usage = F.softmax(self.raw_usage.detach(), 0)
        usage = usage.cpu().numpy().tolist()
        raw_sharpness = self.raw_sharpness.item()
        sharpness = self.raw_sharpness.exp().item()
        return {
            'raw_usage': raw_usage,
            'usage': usage,
            'raw_sharpness': raw_sharpness,
            'sharpness': sharpness,
        }
