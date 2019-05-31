from .....reduce.simple.mean import reduce_mean
from .base import SimpleReduceFork


class Mean(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_mean, *paths)
