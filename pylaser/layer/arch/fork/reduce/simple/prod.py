from .....reduce.simple.prod import reduce_prod
from .base import SimpleReduceFork


class Prod(SimpleReduceFork):
    def __init__(self, *paths):
        super().__init__(reduce_prod, *paths)
