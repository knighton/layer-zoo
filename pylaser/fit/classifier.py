import json
import numpy as np
import os
from time import time
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..util.iteration import each_succ_from_to, each_item
from ..util.summary_stats import summarize_sorted_numpy


class EpochModeResults(object):
    def __init__(self, is_training):
        self.is_training = is_training

        self.losses = []
        self.accuracies = []
        self.tt_forward = []
        self.tt_backward = []

        self.has_summary = False

        self.loss = None
        self.accuracy = None
        self.t_forward = None
        self.t_backward = None

    def update(self, loss, accuracy, t_forward, t_backward=0):
        self.has_summary = False
        self.losses.append(loss)
        self.accuracies.append(accuracy)
        self.tt_forward.append(t_forward)
        self.tt_backward.append(t_backward)

    def done(self):
        go = lambda x: summarize_sorted_numpy(sorted(x))
        self.loss = go(self.losses)
        self.accuracy = go(self.accuracies)
        self.t_forward = go(self.tt_forward)
        self.t_backward = go(self.tt_backward)
        self.has_summary = True

    def summarize(self):
        if not self.has_summary:
            self.done()
        return {
            'loss': self.loss,
            'accuracy': self.accuracy,
            't_forward': self.t_forward,
            't_backward': self.t_backward,
        }


def compute_accuracy(y_pred, y_true):
    y_pred_classes = y_pred.max(1)[1]
    return (y_pred_classes == y_true).type(torch.float32).mean().item()


def train_on_batch(model, x, y_true, optimizer, results=None):
    optimizer.zero_grad()

    model.extra_losses = []

    t = time()
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    t_forward = time() - t

    if hasattr(model, 'extra_losses'):
        for extra_loss in model.extra_losses:
            loss = loss + extra_loss

    t = time()
    loss.backward()
    t_backward = time() - t

    optimizer.step()

    accuracy = compute_accuracy(y_pred, y_true)

    if results:
        loss = loss.item()
        results.update(loss, accuracy, t_forward, t_backward)


def validate_on_batch(model, x, y_true, results=None):
    model.extra_losses = []

    t = time()
    y_pred = model(x)
    loss = F.cross_entropy(y_pred, y_true)
    t_forward = time() - t

    if hasattr(model, 'extra_losses'):
        for extra_loss in model.extra_losses:
            loss = loss + extra_loss

    accuracy = compute_accuracy(y_pred, y_true)

    if results:
        loss = loss.item()
        results.update(loss, accuracy, t_forward)


class EpochResults(object):
    def __init__(self):
        self.train = EpochModeResults(True)
        self.val = EpochModeResults(False)

    def summarize(self):
        return {
            'train': self.train.summarize(),
            'val': self.val.summarize(),
        }


def compute_batches_per_epoch(batch_loader, max_batches_per_epoch=None):
    batches_per_epoch = len(batch_loader)
    if max_batches_per_epoch:
        batches_per_epoch = min(batches_per_epoch, max_batches_per_epoch)
    return batches_per_epoch


def shuffle_epoch(train_loader, val_loader=None,
                  max_train_batches_per_epoch=None,
                  max_val_batches_per_epoch=None):
    train_batches_per_epoch = compute_batches_per_epoch(
        train_loader, max_train_batches_per_epoch)

    if val_loader:
        val_batches_per_epoch = compute_batches_per_epoch(
            val_loader, max_val_batches_per_epoch)
    else:
        val_batches_per_epoch = 0

    modes = [1] * train_batches_per_epoch + [0] * val_batches_per_epoch
    np.random.shuffle(modes)

    return modes


def get_next_sample(each_mode_batch, use_cuda):
    x, y_true = next(each_mode_batch)
    if use_cuda:
        x = x.cuda()
        y_true = y_true.cuda()
    return x, y_true


def each_batch(train_loader, val_loader=None, max_train_batches_per_epoch=None,
               max_val_batches_per_epoch=None, use_cuda=True, use_tqdm=True):
    each_train_batch = each_item(train_loader)

    if val_loader:
        each_val_batch = each_item(val_loader)
    else:
        each_val_batch = None

    modes = shuffle_epoch(train_loader, val_loader, max_train_batches_per_epoch,
                          max_val_batches_per_epoch)
    if use_tqdm:
        modes = tqdm(modes, leave=False)

    for batch_id, is_training in enumerate(modes):
        if is_training:
            each_mode_batch = each_train_batch
        else:
            each_mode_batch = each_val_batch
        x, y_true = get_next_sample(each_mode_batch, use_cuda)
        yield batch_id, is_training, x, y_true


def fit_on_epoch(model, optimizer, train_loader, val_loader=None,
                 max_train_batches_per_epoch=None,
                 max_val_batches_per_epoch=None, use_cuda=True, use_tqdm=True):
    results = EpochResults()

    batches = each_batch(
        train_loader, val_loader, max_train_batches_per_epoch,
        max_val_batches_per_epoch, use_cuda, use_tqdm)

    for batch_id, is_training, x, y_true in batches:
        if is_training:
            model.train()
            train_on_batch(model, x, y_true, optimizer, results.train)
        else:
            model.eval()
            validate_on_batch(model, x, y_true, results.val)

    return results.summarize()


def epoch_results_to_line(epoch_id, x):
    t_acc = x['train']['accuracy']['mean'] * 100
    v_acc = x['val']['accuracy']['mean'] * 100
    return '%6d  %4.1f  %4.1f  %4.1f' % (epoch_id, t_acc, v_acc, t_acc - v_acc)


def fit(model, train_dataset, val_dataset=None, optimizer=None, epochs=None,
        train_batches_per_epoch=None, val_batches_per_epoch=None,
        batch_size=16, data_loader_workers=2, cuda=True, tqdm=True, log=None,
        force=False):
    if force:
        if os.path.exists(log):
            os.remove(log)
    else:
        assert not os.path.exists(log)

    if not optimizer:
        optimizer = Adam

    if cuda:
        model.cuda()

    optimizer = optimizer(model.parameters())

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=data_loader_workers)
    if val_dataset:
        val_loader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=data_loader_workers)
    else:
        val_loader = None

    log_filename = log
    for epoch_id in each_succ_from_to(0, epochs):
        info = fit_on_epoch(model, optimizer, train_loader, val_loader,
                            train_batches_per_epoch, val_batches_per_epoch,
                            cuda, tqdm)
        line = epoch_results_to_line(epoch_id, info)
        print(line)

        if log_filename:
            x = model.summarize()
            log_file = open(log_filename, 'a')
            line = '%s\n' % json.dumps(x, sort_keys=True)
            log_file.write(line)
            log_file.close()
