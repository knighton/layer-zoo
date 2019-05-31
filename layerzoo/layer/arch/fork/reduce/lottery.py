import torch
from torch import nn
from torch.nn import functional as F

from ....reduce.lottery import ReduceLottery
from .base import Fork


class Lottery(Fork):
    def __init__(self, *paths, sharpness=10):
        super().__init__(*paths)
        self.lottery = ReduceLottery(0, len(paths), False, sharpness)

    def forward_inner(self, x):
        yy = self.forward_paths(x)
        y = torch.stack(yy, 0)
        return self.lottery(y)

    def summarize_inner(self, num_percentiles=20):
        paths = self.summarize_paths(num_percentiles)
        lottery = self.lottery.summarize(num_percentiles)
        return {
            'paths': paths,
            'lottery': lottery,
        }
