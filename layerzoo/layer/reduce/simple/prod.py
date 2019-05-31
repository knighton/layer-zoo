from .base import SimpleReduce


def reduce_prod(x, axis, keepdim=False):
    return x.prod(axis, keepdim)


class ReduceProd(SimpleReduce):
    def __init__(self, axis, keepdim=False):
        super().__init__(reduce_prod, axis, keepdim)
