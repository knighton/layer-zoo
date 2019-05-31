from torch import nn
  
from ...base import Layer, summarize


class Flow(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward_inner(self, x):
        return self.layers(x)

    def summarize_layers(self, num_percentiles=20):
        xx = []
        for layer in self.layers:
            x = summarize(layer, num_percentiles)
            xx.append(x)
        return xx

    def summarize_inner(self, num_percentiles=20):
        layers = self.summarize_layers(num_percentiles)
        return {
            'layers': layers,
        }
