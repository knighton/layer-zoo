from ..base import Layer


class Reshape(Layer):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward_inner(self, x):
        batch_shape = (x.shape[0],) + self.shape
        return x.view(*batch_shape)

    def summarize_inner(self, num_percentiles=20):
        return {
            'shape': self.shape,
        }
