import numpy as np
import torch
from torch import nn

from .base import Layer


class Knot(Layer):
    def __init__(self, dim, conn_dim=64):
        super().__init__()

        self.dim = dim
        self.conn_dim = conn_dim

        self.register_buffer('each_unit', torch.arange(dim))

        x = torch.randint(0, dim, (conn_dim, dim))
        self.register_buffer('connections', x)

        self.a = nn.Parameter(torch.randn(conn_dim, dim).cuda())
        self.b = nn.Parameter(torch.randn(conn_dim, conn_dim, dim).cuda())
        self.c = nn.Parameter(torch.randn(conn_dim, dim).cuda())

        self.d = nn.Parameter(torch.randn(conn_dim, dim).cuda())
        self.e = nn.Parameter(torch.randn(conn_dim, conn_dim // 16, dim).cuda())
        self.f = nn.Parameter(torch.randn(conn_dim // 16, dim).cuda())

        self.g = nn.Parameter(torch.randn(conn_dim // 16, dim).cuda())
        self.h = nn.Parameter(torch.randn(conn_dim // 16, 1, dim).cuda())
        self.i = nn.Parameter(torch.randn(1, dim).cuda())

    def sample_connections(self):
        indices = torch.randint(0, self.conn_dim, (self.dim,))
        if self.connections.is_cuda:
            indices = indices.cuda()
        return indices

    def tighten_connections(self):
        root_units = self.each_unit
        indices = self.sample_connections()
        units = self.connections[indices, root_units]
        indices = self.sample_connections()
        units = self.connections[indices, units]
        self.connections[indices, units] = root_units

    def predict(self, x):
        x = torch.einsum('ncw,cw,cdw,dw->ndw', [x, self.a, self.b, self.c])
        x = x - x.mean()
        x = x / x.std()
        x = x.clamp(min=0)
        x = torch.einsum('ncw,cw,cdw,dw->ndw', [x, self.d, self.e, self.f])
        x = x - x.mean()
        x = x / x.std()
        x = x.clamp(min=0)
        return torch.einsum('ncw,cw,cdw,dw->nw', [x, self.g, self.h, self.i])

    def forward_inner(self, x):
        if self.training and np.random.uniform() < 0.25:
            self.tighten_connections()
        pred_in = x.index_select(1, self.connections.view(-1))
        pred_in = pred_in.view(-1, self.conn_dim, self.dim)
        return self.predict(pred_in)
