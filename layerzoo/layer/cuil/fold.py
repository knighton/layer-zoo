from torch.nn import functional as F

from ..base import Layer


class CuilFold2d(Layer):
    """
    Fold conv patches into their image tensor (with cuil dimension).

        Input:  (batch size, cuils, channels, kernel height, kernel width,
                 height, width)
        Output: (batch size, cuils, channels, height, width)
    """

    def __init__(self):
        super().__init__()

    def forward_inner(self, x):
        batch_size, cuils, channels, kernel_height, kernel_width, height, \
            width = x.shape
        assert kernel_height % 2
        assert 1 <= kernel_height
        assert kernel_width % 2
        assert 1 <= kernel_width
        output_size = height, width
        kernel_size = kernel_height, kernel_width
        padding = kernel_height // 2, kernel_width // 2
        x = x.view(batch_size, -1, height * width)
        x = F.fold(x, output_size, kernel_size, 1, padding, 1)
        return x.view(batch_size, cuils, channels, height, width)

    def summarize(self, num_percentiles=20):
        return {}
