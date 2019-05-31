import numpy as np
import torch
from torch import nn

from ..base import Layer
from .rgb_to_hsv import compute_rgb_to_hsv


class RGBToSuperHSV(Layer):
    def __init__(self, num_hue_rotations=16):
        super().__init__()
        self.in_channels = 3
        self.out_channels = num_hue_rotations * 3
        self.num_hue_rotations = num_hue_rotations
        self.hue_rotations = nn.Parameter(torch.rand(num_hue_rotations))
        self.pre_add = nn.Parameter(torch.rand(1, num_hue_rotations))
        self.mul = nn.Parameter(torch.rand(1, num_hue_rotations))
        self.post_add = nn.Parameter(torch.rand(1, num_hue_rotations))

    def forward_inner(self, x):
        go = lambda q: (q.mean().item(), q.std().item(), q.min().item(),
                        q.max().item())

        shape = [1] * len(x.shape)
        shape[1] = -1
        hue_rotations = self.hue_rotations.view(*shape)
        pre_add = self.pre_add.view(*shape)
        mul = self.mul.view(*shape)
        post_add = self.post_add.view(*shape)

        hsv = compute_rgb_to_hsv(x)

        hue, saturation, value = hsv.split(1, 1)

        z = value * 2 - 1
        z = (z + pre_add) * mul + post_add

        rot_hue = (hue + hue_rotations) % 1
        y = (rot_hue * 2 * np.pi).sin() * saturation
        x = (rot_hue * 2 * np.pi).cos() * saturation

        return torch.cat([z, y, x], 1)
