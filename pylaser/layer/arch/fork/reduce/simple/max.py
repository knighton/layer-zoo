from .....reduce.simple.max import reduce_max
from .base import SimpleReduceFork


class Max(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_max, *paths)
