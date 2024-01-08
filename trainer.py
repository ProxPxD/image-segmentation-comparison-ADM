import os
from collections import Counter
from typing import Optional, Callable

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 writer: SummaryWriter,
                 *,
                 model_name: str = 'model',
                 epochs: int = 5,
                 get_model_path: Callable[[str, int, int], str] = None,
                 save_every_n_epoch: int = 1,
                 save_every_n_iteration: int = None,
                 optimizer = None,
                 metrics = None,
                 loss = None,
                 logger = print,
                 device: str = None,
                 verbose: Optional[int] = 1,
                **kwargs
        ):
        self.model: nn.Module = model
        self.model_name = model_name
        self.epochs: int = epochs
        self.get_model_path: Callable[[str, int, int], str] = get_model_path
        self.save_every_n_epoch: int = save_every_n_epoch
        self.save_every_n_iteration: int = save_every_n_iteration
        self.optimizer = optimizer
        self.loss = loss
        self.metrics: dict = metrics
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self.verbose: int = verbose or 0
        self.logger: Callable[[str], None] = logger
        self.counter: Counter = Counter()
        self.writer = writer

        self.epoch: Optional[int] = None
        self.iteration: Optional[int] = None

    def _is_in_right_iteration(self):
        return self.iteration % 1 == 0  # to implement if needed a variable iteration

    def _verbosely_print(self, required_verbose, to_print, and_cond=lambda: True):
        if self.verbose >= required_verbose and and_cond():
            self.logger(to_print)

    def _count(self, full_label: str, to_write):
        self.counter[full_label] += 1
        self.writer.add_scalar(full_label, to_write, self.counter.get(full_label))

    def train(self, train: torch.utils.data.DataLoader, test: torch.utils.data.DataLoader, validation: torch.utils.data.DataLoader = None, epochs: int = None, verbose: int = None):
        print(f'Starting running on {self.device}')
        if epochs:
            self.epochs = epochs
        if verbose is not None:
            self.verbose = verbose
        for epoch in range(self.epochs):
            self.epoch = epoch
            self._verbosely_print(1, f'---Epoch-{epoch+1}/{self.epochs}----------------------')
            for iteration, (X, results) in enumerate(train):
                self.iteration = iteration
                self._verbosely_print(2, f'Iteration {iteration+1:>3}:', self._is_in_right_iteration)
                preds = self.model(X.to(self.device))
                if self.metrics:
                    self._gather_metrics(results, preds)
                self._backwards(results, preds)
                self._optimize()
                del X; del preds
                torch.cuda.empty_cache()
            if self.should_save():
                self.save()

    def _gather_metrics(self, results, preds):
        for metric_name, metric in self.metrics.items():
            self._verbosely_print(3, f'Calculating {metric_name} for train')
            metric = metric.to(self.device)
            metric_result = metric(preds.to(self.device), results.to(self.device)).item()
            full_label = f'{metric_name} - train'
            self._count(full_label, metric_result)

    def _backwards(self, results, preds):
        self._verbosely_print(3, f'Calculating loss')
        loss_result = self.loss(preds.to(self.device), results.to(self.device))
        full_label = f'Loss - train'
        self._count(full_label, loss_result)
        self._verbosely_print(3, f'{full_label}: {loss_result}')
        loss_result.backward(retain_graph=True)

    def _optimize(self):
        if self.optimizer:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def should_save(self):
        return self.epoch and self.epoch % self.save_every_n_epoch == 0

    def save(self):
        path = self.get_model_path(self.model_name, self.epoch, self.iteration)
        if self.verbose > 2:
            self.logger(f'Saving {path}')
        torch.save(self.model.state_dict(), path)

    def load(self):
        ...

    def load_if_exists(self, path):
        existed = os.path.exists(path)
        if existed:
            if self.verbose:
                self.logger(f'Loading {path}')
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint)
        return existed
