from ..base import Reduce


class SimpleReduce(Reduce):
    def __init__(self, reduce, axis, keepdim=False):
        super().__init__(axis, keepdim)
        self.reduce = reduce

    def forward_inner(self, x):
        return self.reduce(x, self.axis, self.keepdim)
