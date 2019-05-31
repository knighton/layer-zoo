from .base import SimpleReduce


def reduce_min(x, axis, keepdim=False):
    return x.min(axis, keepdim)[0]


class ReduceMin(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_min, axis, keepdim)
