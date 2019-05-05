from .base import SimpleReduce


def reduce_sum(x, axis, keepdim=False):
    return x.sum(axis, keepdim)


class ReduceSum(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_sum, axis, keepdim)
