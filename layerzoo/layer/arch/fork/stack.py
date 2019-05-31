import torch

from .base import Fork


class Stack(Fork):
    def __init__(self, *paths, axis=1):
        super().__init__(*paths)
        self.axis = axis

    def forward_inner(self, x):
        yy = self.forward_paths(x)
        return torch.stack(yy, self.axis)

    def summarize_inner(self, num_percentiles=20):
        paths = self.summarize_paths(num_percentiles)
        return {
            'paths': paths,
            'axis': self.axis,
        }
