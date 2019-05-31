from .base import SimpleReduce


def reduce_mean(x, axis, keepdim=False):
    return x.mean(axis, keepdim)


class ReduceMean(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_mean, axis, keepdim)
