from ..base import Layer


class Print(Layer):
    def __init__(self, name=None):
        super().__init__()
        self.name = name
        self.has_run = False

    def forward_inner(self, x):
        if self.has_run:
            return x
        name = self.name or ''
        print('%s :: %s' % (name, x.shape))
        self.has_run = True
        return x

    def summarize_inner(self, num_percentiles=20):
        return {
            'name': self.name,
        }
