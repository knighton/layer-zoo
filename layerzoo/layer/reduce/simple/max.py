from .base import SimpleReduce


def reduce_max(x, axis, keepdim=False):
    return x.max(axis, keepdim)[0]


class ReduceMax(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_max, axis, keepdim)
