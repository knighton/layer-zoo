from ...base import Layer, summarize


class Fork(Layer):
    def __init__(self, *paths):
        super().__init__()
        self.paths = []
        for i, path in enumerate(paths):
            self.paths.append(path)
            setattr(self, 'path_%d' % i, path)

    def forward_paths(self, x):
        yy = []
        for path in self.paths:
            y = path(x)
            yy.append(y)
        return yy

    def forward_inner(self, x):
        raise NotImplementedError

    def summarize_paths(self, num_percentiles=20):
        xx = []
        for path in self.paths:
            x = summarize(path, num_percentiles)
            xx.append(x)
        return xx

    def summarize_inner(self, num_percentiles=20):
        raise NotImplementedError
