import torch
from torch import nn
from torch.nn import functional as F

from ....reduce.choose import ReduceChoose
from .base import Fork


class Choose(Fork):
    def __init__(self, *paths):
        super().__init__(*paths)
        self.choose = ReduceChoose(0, len(paths))

    def forward_inner(self, x):
        yy = self.forward_paths(x)
        y = torch.stack(yy, 0)
        return self.choose(y)

    def summarize_inner(self, num_percentiles=20):
        paths = self.summarize_paths(num_percentiles)
        choose = self.choose.summarize(num_percentiles)
        return {
            'paths': paths,
            'choose': choose,
        }
