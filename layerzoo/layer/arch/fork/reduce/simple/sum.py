from .....reduce.simple.sum import reduce_sum
from .base import SimpleReduceFork


class Sum(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_sum, *paths)
