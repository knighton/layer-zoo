import torch
from torch import nn
from torch.nn import functional as F

from .base import Fork


class Choose(Fork):
    def __init__(self, *paths):
        super().__init__(*paths)
        x = torch.zeros(len(paths))
        self.raw_usage = nn.Parameter(x)

    def forward_inner(self, x):
        usage = F.softmax(self.raw_usage, 0)
        shape = (-1,) + (1,) * len(x.shape)
        usage = usage.view(*shape)
        yy = self.forward_paths(x)
        y = torch.stack(yy, 0)
        return (usage * y).sum(0)

    def summarize_inner(self, num_percentiles=20):
        paths = self.summarize_paths(num_percentiles)
        raw_usage = self.raw_usage.detach().cpu().numpy().tolist()
        usage = F.softmax(self.raw_usage.detach(), 0)
        usage = usage.cpu().numpy().tolist()
        return {
            'raw_uage': raw_usage,
            'usage': usage,
            'paths': paths,
        }
