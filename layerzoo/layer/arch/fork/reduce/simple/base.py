from ..base import ReduceFork


class SimpleReduceFork(ReduceFork):
    def __init__(self, reduce, *paths):
        super().__init__(*paths)
        self.reduce = reduce

    def forward_inner(self, x):
        yy = self.forward_paths(x)
        y = torch.stack(yy, 0)
        return self.reduce(y, 0)

    def summarize_inner(self, num_percentiles=20):
        paths = self.summarize_paths(num_percentiles)
        return {
            'paths': paths,
        }
