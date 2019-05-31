import numpy as np
from time import time
import torch
from torch import nn

from ..util.moving_stats import MovingStatistics


dtype_to_str = lambda x: str(x).replace('torch.', '')


class TensorSpec(object):
    @classmethod
    def from_tensor(cls, x):
        return cls(x.shape[1:], x.dtype)

    def __init__(self, shape, dtype):
        self.shape = shape
        self.dtype = dtype

    def accepts(self, x):
        return x.shape[1:] == self.shape and x.dtype == self.dtype

    def summarize(self):
        return {
            'shape': self.shape,
            'dtype': dtype_to_str(self.dtype),
        }


class Layer(nn.Module):
    def __init__(self, time_stats_size=1000, time_stats_rate=0.01,
                 time_stats_dtype=np.float32):
        super().__init__()
        self._layerzoo_in_spec = None
        self._layerzoo_out_spec = None
        new = lambda: MovingStatistics(time_stats_size, time_stats_rate,
                                       time_stats_dtype)
        self._layerzoo_train_time_stats = new()
        self._layerzoo_val_time_stats = new()

    def forward_inner(self, x):
        return x

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            if self._layerzoo_in_spec is None:
                self._layerzoo_in_spec = TensorSpec.from_tensor(x)
            else:
                assert self._layerzoo_in_spec.accepts(x)
        else:
            assert self._layerzoo_in_spec is None

        t = time()
        y = self.forward_inner(x)
        t = time() - t

        if self.training:
            time_stats = self._layerzoo_train_time_stats
        else:
            time_stats = self._layerzoo_val_time_stats
        time_stats.update(t)

        if isinstance(y, torch.Tensor):
            if self._layerzoo_out_spec is None:
                self._layerzoo_out_spec = TensorSpec.from_tensor(y)
            else:
                assert self._layerzoo_out_spec.accepts(y)
        else:
            assert self._layerzoo_out_spec is None

        return y

    def summarize_inner(self, num_percentiles=20):
        return None

    def summarize(self, num_percentiles=20):
        if self._layerzoo_in_spec:
            in_spec = self._layerzoo_in_spec.summarize()
        else:
            in_spec = None
        if self._layerzoo_out_spec:
            out_spec = self._layerzoo_out_spec.summarize()
        else:
            out_spec = None
        spec = {
            'in': in_spec,
            'out': out_spec,
        }

        train_time = self._layerzoo_train_time_stats.summarize(num_percentiles)
        val_time = self._layerzoo_val_time_stats.summarize(num_percentiles)
        time = {
            'train': train_time,
            'val': val_time,
        }

        body = self.summarize_inner(num_percentiles)

        return {
            'type': self.__class__.__name__,
            'spec': spec,
            'time': time,
            'body': body,
        }


def summarize(layer, num_percentiles=20):
    if isinstance(layer, Layer):
        return layer.summarize(num_percentiles)
    elif isinstance(layer, nn.Module):
        return layer.__class__.__name__
    else:
        assert False
