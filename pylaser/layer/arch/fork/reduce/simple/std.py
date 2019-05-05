from .....reduce.simple.std import reduce_std
from .base import SimpleReduceFork


class Std(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_std, *paths)
