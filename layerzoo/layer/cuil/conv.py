from ...util.layer import normalize_coords
from ..base import Layer
from .convolve import CuilConvolve2d
from .dense import CuilDense


class CuilConv2dInner(Layer):
    def __init__(self, cuils, in_channels, out_channels, kernel_size=3):
        super().__init__()
        kernel_height, kernel_width = normalize_coords(kernel_size, 2)
        in_dim = in_channels * kernel_height * kernel_width
        out_dim = out_channels
        self.cuil_dense = CuilDense(cuils, in_dim, out_dim)

    def forward_inner(self, x):
        batch_size, cuils, in_channels, kernel_height, kernel_width = x.shape
        x = x.view(batch_size, cuils, -1)
        return self.cuil_dense(x)

    def summarize_inner(self, num_percentiles=20):
        return {
            'cuil_dense': self.cuil_dense.summarize(num_percentiles)
        }
        

class CuilConv2d(CuilConvolve2d):
    def __init__(self, cuils, in_channels, out_channels, kernel_size=3):
        kernel_size = kernel_height, kernel_width = \
            normalize_coords(kernel_size, 2)
        inner = CuilConv2dInner(cuils, in_channels, out_channels, kernel_size)
        super().__init__(inner, kernel_size)
