from torch import nn

from pylaser.layer import *


class Classifier2d(Sequence):
    pass


class Conv2dBlock(Sequence):
    def __init__(self, channels, height, width):
        super().__init__(
            Append(Scatter2d(channels, channels, height, width)),
            Hinge(),
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )


class Conv2dBlock(Sequence):
    def __init__(self, channels, height, width):
        one = Sequence(
            Hinge(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

        two = Sequence(
            Hinge(),
            nn.Conv2d(channels, channels // 8, 1, 1, 0),
            nn.BatchNorm2d(channels // 8),
            Hinge(),
            nn.Conv2d(channels // 8, channels, 5, 1, 2),
            nn.BatchNorm2d(channels),
        )

        choose = Choose(one, two)

        super().__init__(choose)


class Conv2dBlock(Sequence):
    def __init__(self, channels, height, width):
        super().__init__(
            Hinge(),
            nn.Conv2d(channels, channels // 8, 1, 1, 0),
            nn.BatchNorm2d(channels // 8),

            Append(SkewedSample(channels // 8, channels // 8, 2)),

            Hinge(),
            nn.Conv2d(channels // 4, channels, 5, 1, 2),
            nn.BatchNorm2d(channels),
        )


class Conv2dBlock(Sequence):
    def __init__(self, channels, height, width):
        super().__init__(
            Hinge(),
            nn.Conv2d(channels, channels // 8, 1, 1, 0),
            nn.BatchNorm2d(channels // 8),
            Hinge(),
            nn.Conv2d(channels // 8, channels, 5, 1, 2),
            nn.BatchNorm2d(channels),
        )


class DenseBlock(Sequence):
    def __init__(self, dim):
        super().__init__(
            Hinge(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )


class DenseBlock(Sequence):
    def __init__(self, dim):
        dense = Sequence(
            Hinge(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

        describe = Sequence(
            Describe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )

        shuffled = Choose(
            WhileShuffled(dim, dim * 4, Sequence(
                Hinge(),
                nn.Dropout(),
                ShardedDense(dim * 4, dim * 4, dim),
                nn.BatchNorm1d(dim * 4),
            )),
            WhileShuffled(dim, dim * 4, Sequence(
                Hinge(),
                nn.Dropout(),
                ShardedDense(dim * 4, dim * 4, dim),
                nn.BatchNorm1d(dim * 4),
            )),
        )

        choose = Choose(dense, describe, shuffled)

        super().__init__(choose)


class DenseBlock(Sequence):
    def __init__(self, dim):
        super().__init__(
            Describe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )


class BaselineClassifier2d(Classifier2d):
    def __init__(self, in_channels, out_dim, channels):
        super().__init__(
            nn.Conv2d(in_channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),

            Skip(Conv2dBlock(channels, 32, 32)),
            Skip(Conv2dBlock(channels, 32, 32)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels, 16, 16)),
            Skip(Conv2dBlock(channels, 16, 16)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels, 8, 8)),
            Skip(Conv2dBlock(channels, 8, 8)),

            nn.MaxPool2d(2),

            Skip(Conv2dBlock(channels, 4, 4)),
            Skip(Conv2dBlock(channels, 4, 4)),

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

            Hinge(),
            nn.Dropout(),
            nn.Linear(channels, out_dim),
        )
