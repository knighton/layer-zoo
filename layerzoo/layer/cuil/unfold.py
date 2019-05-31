from torch.nn import functional as F

from ...util.layer import normalize_coords
from ..base import Layer


class CuilUnfold2d(Layer):
    """
    Unfold an image tensor into its conv patches (with cuil dimension).

        Input:  (batch size, cuils, channels, height, width)
        Output: (batch size, cuils, channels, kernel height, kernel width,
                 height, width)
    """

    def __init__(self, kernel_size):
        super().__init__()
        kernel_size = kernel_height, kernel_width = \
            normalize_coords(kernel_size, 2)
        assert 1 <= kernel_size[0]
        assert kernel_size[0] % 2
        assert 1 <= kernel_size[1]
        assert kernel_size[1] % 2
        self.kernel_size = kernel_size
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.padding = kernel_height // 2, kernel_width // 2

    def forward_inner(self, x):
        batch_size, cuils, channels, height, width = x.shape
        x = x.view(batch_size, -1, height, width)
        x = F.unfold(x, self.kernel_size, 1, self.padding, 1)
        return x.view(batch_size, cuils, channels, self.kernel_height,
                      self.kernel_width, height, width)

    def summarize(self, num_percentiles=20):
        return {
            'kernel_size': self.kernel_size,
            'padding': self.padding,
        }
