from .....reduce.simple.min import reduce_min
from .base import SimpleReduceFork


class Min(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_min, *paths)
