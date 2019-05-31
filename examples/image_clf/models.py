from torch import nn

from layerzoo.layer import *


class Classifier2d(Flow):
    pass


class Conv2dBlock(Flow):
    def __init__(self, channels, height, width):
        super().__init__(
            Append(Scatter2d(channels, channels, height, width)),
            Hinge(),
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )


class Conv2dBlock(Flow):
    def __init__(self, channels, height, width):
        one = Flow(
            Hinge(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

        two = Flow(
            Hinge(),
            nn.Conv2d(channels, channels // 8, 1, 1, 0),
            nn.BatchNorm2d(channels // 8),
            Hinge(),
            nn.Conv2d(channels // 8, channels, 5, 1, 2),
            nn.BatchNorm2d(channels),
        )

        choose = Choose(one, two)

        super().__init__(choose)


class Conv2dBlock(Flow):
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


class Conv2dBlock(Flow):
    def __init__(self, channels, height, width):
        super().__init__(
            Hinge(),
            nn.Conv2d(channels, channels // 8, 1, 1, 0),
            nn.BatchNorm2d(channels // 8),
            Hinge(),
            nn.Conv2d(channels // 8, channels, 5, 1, 2),
            nn.BatchNorm2d(channels),
        )


class DenseBlock(Flow):
    def __init__(self, dim):
        super().__init__(
            Hinge(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )


class DenseBlock(Flow):
    def __init__(self, dim):
        dense = Flow(
            Hinge(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

        describe = Flow(
            Describe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )

        shuffled = Choose(
            WhileShuffled(dim, dim * 4, Flow(
                Hinge(),
                nn.Dropout(),
                ShardedDense(dim * 4, dim * 4, dim),
                nn.BatchNorm1d(dim * 4),
            )),
            WhileShuffled(dim, dim * 4, Flow(
                Hinge(),
                nn.Dropout(),
                ShardedDense(dim * 4, dim * 4, dim),
                nn.BatchNorm1d(dim * 4),
            )),
        )

        choose = Choose(dense, describe, shuffled)

        super().__init__(choose)


class DenseBlock(Flow):
    def __init__(self, dim):
        linear = Flow(
            Describe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )

        choose = Flow(
            Describe(dim, dim * 4),
            Reshape(4, dim),
            ReduceChoose(1, 4),
            nn.BatchNorm1d(dim),
        )

        lottery = Flow(
            Describe(dim, dim * 4),
            Reshape(4, dim),
            ReduceLottery(1, 4),
            nn.BatchNorm1d(dim),
        )

        choose = Choose(linear, choose, lottery)

        super().__init__(choose)


class DenseBlock(Flow):
    def __init__(self, dim):
        static = Flow(
            Describe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )

        dynamic = Flow(
            DynamicDescribe(dim, dim * 4),
            nn.Linear(dim * 4, dim),
            nn.BatchNorm1d(dim),
        )

        choose = Choose(static, dynamic)

        super().__init__(choose)


class DenseBlock(Flow):
    def __init__(self, dim):
        dense = Flow(
            Hinge(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

        compute_rate = Flow(
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        dropped = Flow(
            Hinge(),
            DynamicDropout(compute_rate),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )

        choose = Choose(dense, dropped)

        super().__init__(choose)


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


class BaselineClassifier2d(Classifier2d):
    @classmethod
    def new_inner(cls, in_dim, out_dim):
        mid_dim = (in_dim + out_dim) // 2
        return Flow(
            Describe(in_dim, in_dim),
            nn.Linear(in_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def __init__(self, in_channels, out_dim, channels):
        num_stages = 8
        inner_out_dim = channels
        accumulate = Accumulate(num_stages, self.new_inner, channels * 16,
                                inner_out_dim)

        mid_dim = accumulate.out_dim // 4
        tail = Flow(
            Hinge(),
            nn.Dropout(),
            nn.Linear(accumulate.out_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, channels),
            nn.BatchNorm1d(channels),
            Hinge(),
            nn.Dropout(),
            nn.Linear(channels, out_dim),
        )

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

            accumulate,

            tail,
        )


class BaselineClassifier2d(Classifier2d):
    @classmethod
    def new_inner(cls, in_dim, out_dim):
        mid_dim = (in_dim + out_dim) // 2
        return Flow(
            Sample(in_dim, in_dim // 4),
            Describe(in_dim // 4, in_dim * 4),
            nn.Linear(in_dim * 4, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def __init__(self, in_channels, out_dim, channels):
        num_stages = 16
        inner_out_dim = channels // 2
        accumulate = Accumulate(num_stages, self.new_inner, channels * 16,
                                inner_out_dim)

        mid_dim = accumulate.out_dim // 4
        tail = Flow(
            Hinge(),
            nn.Dropout(),
            nn.Linear(accumulate.out_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, channels),
            nn.BatchNorm1d(channels),
            Hinge(),
            nn.Dropout(),
            nn.Linear(channels, out_dim),
        )

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

            accumulate,

            tail,
        )


class BaselineClassifier2d(Classifier2d):
    @classmethod
    def new_inner(cls, in_dim, out_dim):
        mid_dim = (in_dim + out_dim) // 2
        return Flow(
            Describe(in_dim, in_dim * 4),
            nn.Linear(in_dim * 4, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def __init__(self, in_channels, out_dim, channels):
        num_stages = 4
        inner_out_dim = channels
        percolate = Percolate(num_stages, self.new_inner, channels * 16,
                              inner_out_dim)

        mid_dim = percolate.out_dim // 4
        tail = Flow(
            Hinge(),
            nn.Dropout(),
            nn.Linear(percolate.out_dim, mid_dim),
            nn.BatchNorm1d(mid_dim),
            Hinge(),
            nn.Dropout(),
            nn.Linear(mid_dim, out_dim),
        )

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

            percolate,

            tail,
        )
