from torch import nn

from pylaser.layer import *


class Classifier2d(Sequence):
    pass


class Conv2dBlock(Sequence):
    def __init__(self, channels):
        super().__init__(
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )


class DenseBlock(Sequence):
    def __init__(self, dim):
        super().__init__(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )


class BaselineClassifier2d(Classifier2d):
    def __init__(self, in_channels, out_dim, channels):
        super().__init__(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),

            Skip(Conv2dBlock(channels)),
            Skip(Conv2dBlock(channels)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels)),
            Skip(Conv2dBlock(channels)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels)),
            Skip(Conv2dBlock(channels)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels)),
            Skip(Conv2dBlock(channels)),

            Flatten(),

            Skip(DenseBlock(channels * 16)),
            Skip(DenseBlock(channels * 16)),

            Reshape(4, -1),
            nn.MaxPool1d(4),
            Flatten(),

            Skip(DenseBlock(channels * 4)),
            Skip(DenseBlock(channels * 4)),

            Reshape(4, -1),
            nn.MaxPool1d(4),
            Flatten(),

            Skip(DenseBlock(channels)),
            Skip(DenseBlock(channels)),

            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(channels, out_dim),
        )
