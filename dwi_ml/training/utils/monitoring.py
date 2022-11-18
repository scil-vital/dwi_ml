# -*- coding: utf-8 -*-
from collections import deque
import timeit
from datetime import datetime

import numpy as np


class TimeMonitor(object):
    def __init__(self):
        self.epoch_durations = []
        self._start_time = None

    def start_new_epoch(self):
        self._start_time = datetime.now()

    def end_epoch(self):
        if self._start_time is None:
            raise ValueError("You should not end the epoch; it has not "
                             "started (or you haven't told the TimeMonitor).")
        duration = datetime.now() - self._start_time

        # Saving duration in minutes
        self.epoch_durations.append(duration.total_seconds() / 60)
        self._start_time = None


class ValueHistoryMonitor(object):
    """ History of some value for each iteration during training, and mean
    value for each epoch.

    Example of usage: History of the loss during training.
        loss_monitor = ValueHistoryMonitor()
        ...
        loss_monitor.start_new_epoch()

        # Call update at each iteration
        loss_monitor.update(2.3)
                ...
        loss_monitor.end_epoch()  # call at epoch end
        ...
        loss_monitor.epochs_means  # returns the loss curve as a list
    """

    def __init__(self, name):
        self.name = name
        # self.all_updates_history[i] = values of all batches for epoch i.
        self.all_updates_history = []
        self.epochs_means = []
        self.current_epoch = -1

    def update(self, value):
        """
        Note. Does not save the update if value is inf.

        Parameters
        ----------
        value: The value of the loss
        """
        if np.isinf(value):
            return

        self.all_updates_history[-1].append(value)

    def start_new_epoch(self):
        assert len(self.epochs_means) == self.current_epoch + 1, \
            "Did you forget to end previous epoch? len(epochs_means) is {}" \
            "but current epoch is {}" \
            .format(len(self.epochs_means), self.current_epoch)
        assert len(self.all_updates_history) == self.current_epoch + 1, \
            "Unexpected error. all_updates_history is of len {} but current " \
            "epoch is {}" \
            .format(self.all_updates_history, self.current_epoch)

        self.current_epoch += 1
        self.all_updates_history.append([])

    def end_epoch(self):
        """
        Compute mean of current epoch and add it to means values.
        """
        assert len(self.all_updates_history) == self.current_epoch + 1
        assert len(self.epochs_means) == len(self.all_updates_history) - 1

        self.epochs_means.append(np.mean(self.all_updates_history[-1]))

    def get_state(self):
        return {'name': self.name,
                'all_updates_history': self.all_updates_history,
                'epochs_means': self.epochs_means,
                'current_epoch': self.current_epoch
                }

    def set_state(self, state):
        self.name = state['name']
        self.all_updates_history = state['all_updates_history']
        self.epochs_means = state['epochs_means']
        self.current_epoch = state['current_epoch']


class BestEpochMonitoring(object):
    """
    Object to stop training early if the loss doesn't improve after a given
    number of epochs ("patience").
    """

    def __init__(self, patience: int, min_eps: float = 1e-6):
        """
        Parameters
        -----------
        patience: int
            Maximal number of bad epochs we allow.
        min_eps: float, optional
            Precision term to define what we consider as "improving": when the
            loss is at least min_eps smaller than the previous best loss.
        """
        self.patience = patience
        self.min_eps = min_eps

        self.best_value = None
        self.best_epoch = None
        self.n_bad_epochs = None

    def update(self, loss, epoch):
        """
        Parameters
        ----------
        loss : float
            Loss value for a new training epoch
        epoch : int
            Current epoch

        Returns
        -------
        is_bad: bool
            True if this epoch was a bad epoch.
        """
        if self.best_value is None:
            # First epoch. Setting values.
            self.best_value = loss
            self.best_epoch = epoch
            self.n_bad_epochs = 0
            return False
        elif loss < self.best_value - self.min_eps:
            # Improving from at least eps.
            self.best_value = loss
            self.best_epoch = epoch
            self.n_bad_epochs = 0
            return False
        else:
            # Not improving enough
            self.n_bad_epochs += 1
            return True

    @property
    def is_patience_reached(self):
        """
        Returns
        -------
        True if the number of epochs without improvements is more than the
        patience.
        """
        if self.n_bad_epochs >= self.patience:
            return True

        return False

    def get_state(self):
        """ Get object state """
        return {'patience': self.patience,
                'min_eps': self.min_eps,
                'best_value': self.best_value,
                'best_epoch': self.best_epoch,
                'n_bad_epochs': self.n_bad_epochs}

    def set_state(self, state):
        """ Set object state """
        self.patience = state['patience']
        self.min_eps = state['min_eps']
        self.best_value = state['best_value']
        self.best_epoch = state['best_epoch']
        self.n_bad_epochs = state['n_bad_epochs']


class EarlyStoppingError(Exception):
    """Exception raised when an experiment is stopped by early-stopping

    Attributes
        message -- explanation of why early stopping occured"""

    def __init__(self, message):
        self.message = message


class IterTimer(object):
    """
    Hint: After each iteration, you can check that the maximum allowed time has
    not been reached by using:

    # Ex: To check that time remaining is less than one iter + 30 seconds
    time.time() + iter_timer.mean + 30 > max_time

    # Ex: To allow some incertainty. Ex: prevent continuing in the case the
    # next iter could be twice as long as usual:
    time.time() + iter_timer.mean * 2.0 + 30 > max_time
    """
    def __init__(self, history_len=5):
        self.history = deque(maxlen=history_len)
        self.iterable = None
        self.start_time = None

    def __call__(self, iterable):
        self.iterable = iter(iterable)
        return self

    def __iter__(self):
        return self

    def __next__(self):
        if self.start_time is not None:
            elapsed = timeit.default_timer() - self.start_time
            self.history.append(elapsed)
        self.start_time = timeit.default_timer()
        return next(self.iterable)

    @property
    def mean(self):
        return np.mean(self.history) if len(self.history) > 0 else 0
