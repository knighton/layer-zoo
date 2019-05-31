from ...util.layer import normalize_coords
from ..base import Layer
from .fold import CuilFold2d
from .unfold import CuilUnfold2d


class CuilConvolve2d(Layer):
    def __init__(self, inner, kernel_size=3):
        super().__init__()
        kernel_size = normalize_coords(kernel_size, 2)
        self.unfold = CuilUnfold2d(kernel_size)
        self.inner = inner
        self.fold = CuilFold2d()
        self.kernel_size = kernel_size

    def forward_inner(self, x):
        x = self.unfold(x)
        batch_size, cuils, in_channels, kernel_height, kernel_width, height, \
           width = x.shape
        x = x.permute(0, 5, 6, 1, 2, 3, 4).contiguous()
        x = x.view(-1, cuils, in_channels, kernel_height, kernel_width)
        x = self.inner(x)
        x = x.view(batch_size, height, width, cuils, -1, 1, 1)
        x = x.permute(0, 3, 4, 5, 6, 1, 2).contiguous()
        return self.fold(x)

    def summarize_inner(self, num_percentiles=20):
        return {
            'fold': self.fold.summarize(num_percentiles),
            'inner': self.inner.summarize(num_percentiles),
            'unfold': self.unfold.summarize(num_percentiles),
            'kernel_size': self.kernel_size,
        }
