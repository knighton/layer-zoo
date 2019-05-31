import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ..util.summary_stats import summarize_tensor
from .base import Layer
from .stats.statistics import Statistics


def compute_corrcoef(x):
    batch_size = x.shape[0]
    x = x.t().mm(x)
    x = x / (batch_size - 1)
    std = x.diag().sqrt()
    x = x / std.unsqueeze(0)
    x = x / std.unsqueeze(1)
    return x.clamp(min=-1, max=1)


def als_step_users(user_vecs, item_vecs, ratings, user_reg=0):
    yty = item_vecs.t() @ item_vecs
    lambda_eye = torch.eye(yty.shape[0]) * user_reg
    if user_vecs.is_cuda:
        lambda_eye = lambda_eye.cuda()
    a = yty + lambda_eye
    b = (ratings @ item_vecs).t()
    user_vecs, _ = torch.gesv(b, a)
    return user_vecs.t()


def als_step_items(user_vecs, item_vecs, ratings, item_reg=0):
    xtx = user_vecs.t() @ user_vecs
    lambda_eye = torch.eye(xtx.shape[0]) * item_reg
    if user_vecs.is_cuda:
        lambda_eye = lambda_eye.cuda()
    a = xtx + lambda_eye
    b = user_vecs.t() @ ratings
    item_vecs, _ = torch.gesv(b, a)
    return item_vecs.t()


def als_step(user_vecs, item_vecs, ratings, user_reg=0, item_reg=0):
    user_vecs = als_step_users(user_vecs, item_vecs, ratings, user_reg)
    item_vecs = als_step_items(user_vecs, item_vecs, ratings, item_reg)
    return user_vecs, item_vecs


class Cohere(Layer):
    def __init__(self, model, in_dim, out_dim, embed_dim=128,
                 cache_momentum=0.9, cache_slots=256):
        super().__init__()
        self.models = model,
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.embed_dim = embed_dim
        self.cache = Statistics(in_dim, cache_momentum, cache_slots)
        self.register_buffer('in_embeddings', torch.randn(in_dim, embed_dim))
        self.register_buffer('other_in_embeddings',
                             torch.randn(in_dim, embed_dim))
        self.out_embeddings = nn.Parameter(torch.randn(out_dim, embed_dim))
        self.scale = nn.Parameter(torch.Tensor([1]))

    def forward_inner(self, x):
        x_center = x - x.mean(0)
        self.cache.update(x_center)

        if np.random.uniform() < 0.25:
            correlations = compute_corrcoef(self.cache.samples)

            self.in_embeddings, self.other_in_embeddings = als_step(
                self.in_embeddings, self.other_in_embeddings, correlations)

            if np.random.uniform() < 0.01:
                pred_correlations = \
                    self.in_embeddings @ self.other_in_embeddings.t()
                loss = (correlations - pred_correlations) ** 2
                loss = loss.mean()
                print('::', 'loss', loss.item())

            cor = compute_corrcoef(self.out_embeddings)
            loss = cor.abs().mean() * 10
            model, = self.models
            model.extra_losses.append(loss)

        io = torch.einsum('ie,oe->io',
                          [self.in_embeddings, self.out_embeddings])
        io = self.scale * io
        io = F.softmax(io, 1)
        return torch.einsum('ni,io->no', [x, io])

    def summarize_inner(self, num_percentiles=20):
        cor = torch.einsum('ie,je->ij',
                           [self.in_embeddings, self.other_in_embeddings])
        return {
            'scale': self.scale.item(),
            'correlations:mean': cor.mean().item(),
            'correlations:std': cor.std().item(),
            'correlations:min': cor.min().item(),
            'correlations:max': cor.max().item(),
        }
