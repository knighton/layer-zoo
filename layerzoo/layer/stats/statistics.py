import torch

from ..base import Layer


class Statistics(Layer):
    def __init__(self, dim, momentum=0.95, slots=512):
        super().__init__()
        assert 0 <= momentum < 1
        assert isinstance(slots, int)
        assert 1 <= slots
        self.dim = dim
        self.momentum = momentum
        self.slots = slots
        self.register_buffer('samples', torch.zeros(slots, dim))
        self.updates = 0

    @classmethod
    def compute_churn(cls, momentum, slots, updates):
        update_rate = 1 - momentum
        cross_over = int(update_rate ** -1)
        if updates < cross_over:
            churn = int(slots / (updates + 1))
        else:
            churn = int(slots * update_rate)
        return churn

    def update(self, x):
        x = x.detach()
        churn = self.compute_churn(self.momentum, self.slots, self.updates)
        victim_slots = torch.randperm(self.slots)[:churn]
        samples_to_keep = torch.randint(0, x.shape[0], (churn,))
        self.samples[victim_slots, :] = x[samples_to_keep, :]
        self.updates += 1

    def forward_inner_inner(self, x):
        return x

    def forward_inner(self, x):
        self.update(x)
        return self.forward_inner_inner(x)

    def summarize_inner_inner(self, num_percentiles=20):
        return None

    def summarize_inner(self, num_percentiles=20):
        return {
            'dim': self.dim,
            'momentum': self.momentum,
            'slots': self.slots,
            'inner': self.summarize_inner_inner(num_percentiles),
        }
