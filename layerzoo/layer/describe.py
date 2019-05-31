import torch
from torch import nn
from torch.nn import functional as F

from .base import Layer


class Describe(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        feature_dim = 4

        assert not out_dim % in_dim
        desc_dim = out_dim // in_dim

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.desc_dim = desc_dim
        self.feature_dim = feature_dim

        x = torch.randn(1, desc_dim, in_dim)
        self.target = nn.Parameter(x)

        x = torch.randn(1, desc_dim, in_dim)
        self.sharpness = nn.Parameter(x)

        x = torch.randn(1, feature_dim, desc_dim, in_dim)
        self.feature_weights = nn.Parameter(x)

    def forward_inner(self, x):
        go = lambda x: (x.min().item(), x.mean().item(), x.max().item())

        batch_size = x.shape[0]
        x = x.unsqueeze(1)
        gap = x - self.target
        sharpness = self.sharpness.exp()
        miss = (gap * sharpness).tanh()

        is_near = 1 - miss.abs()
        is_far = 1 - is_near
        is_less = 0.5 - miss / 2
        is_more = 1 - is_less
        features = is_near, is_far, is_less, is_more
        x = torch.stack(features, 1)

        feature_weights = F.softmax(self.feature_weights, 1)
        y = (x * feature_weights).sum(1)

        return y.view(batch_size, -1)

    def summarize(self, num_percentiles=20):
        def go(x):
            x = x.detach()
            mean = x.mean().item()
            std = x.std().item()
            return {
                'mean': mean,
                'std': std,
            }

        target = go(self.target)
        sharpness = go(self.sharpness.exp())
        feature_weights = go(F.softmax(self.feature_weights, 1))

        return {
            'target': target,
            'sharpness': sharpness,
            'feature_weights': feature_weights,
        }
