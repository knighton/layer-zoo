from .base import SimpleReduce


def reduce_std(x, axis, keepdim=False):
    return x.std(axis, keepdim)


class ReduceStd(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_std, axis, keepdim)
