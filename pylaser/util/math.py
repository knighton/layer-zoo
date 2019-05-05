import torch


def inverse_sigmoid(x):
    assert 0 < x < 1
    return -(1 / x - 1).log()


def int_randn(shape, std=1, dtype=torch.int64):
    x = torch.randn(*shape) * std
    x = x + 0.5 - (x < 0).float()
    return x.type(dtype)


def int_randn_nonzero(shape, std=1, dtype=torch.int64):
    x = torch.randn(*shape) * std
    x = x + 1 - 2 * (x < 0).float()
    return x.type(dtype)
