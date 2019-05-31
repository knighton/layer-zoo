from argparse import ArgumentParser
  
from layerzoo import datasets
from layerzoo.fit.classifier import fit
from . import models


def parse_flags():
    a = ArgumentParser()
    a.add_argument('--clf', type=str, default='BaselineClassifier2d:' + \
                   'in_channels=3,out_dim=10,channels=128')
    a.add_argument('--dataset', type=str, default='cifar10')
    a.add_argument('--epochs', type=int, default=-1)
    a.add_argument('--train_batches_per_epoch', type=int, default=100)
    a.add_argument('--val_batches_per_epoch', type=int, default=50)
    a.add_argument('--batch_size', type=int, default=16)
    a.add_argument('--data_loader_workers', type=int, default=2)
    a.add_argument('--cuda', type=int, default=1)
    a.add_argument('--tqdm', type=int, default=1)
    a.add_argument('--log', type=str, default='')
    a.add_argument('--force', type=int, default=0)
    return a.parse_args()


def parse_clf_kwarg_value(s):
    try:
        return int(s)
    except:
        pass
    try:
        return float(s)
    except:
        pass
    return s


def parse_clf_kwargs(s):
    kwargs = {}
    ss = s.split(',')
    for s in ss:
        index = s.index('=')
        k = s[:index]
        v = parse_clf_kwarg_value(s[index + 1:])
        kwargs[k] = v
    return kwargs


def parse_clf(s):
    index = s.find(':')
    if index == -1:
        class_name = s
        kwargs = {}
    else:
        class_name = s[:index]
        kwargs = parse_clf_kwargs(s[index + 1:])
    klass = getattr(models, class_name)
    return klass(**kwargs)


def parse_dataset(s):
    func = getattr(datasets, s)
    return func()


def main(flags):
    clf = parse_clf(flags.clf)
    optimizer = None
    train_dataset, val_dataset = parse_dataset(flags.dataset)
    fit(clf, train_dataset, val_dataset, optimizer, flags.epochs,
        flags.train_batches_per_epoch, flags.val_batches_per_epoch,
        flags.batch_size, flags.data_loader_workers, flags.cuda, flags.tqdm,
        flags.log, flags.force)


if __name__ == '__main__':
    main(parse_flags())
