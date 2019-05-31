from ..base import Layer


class Reduce(Layer):
    def __init__(self, axis, keepdim=False):
        super().__init__()
        self.axis = axis
        self.keepdim = keepdim

    def forward_inner(self, x):
        raise NotImplementedError
