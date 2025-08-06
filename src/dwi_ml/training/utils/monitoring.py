# -*- coding: utf-8 -*-
import logging
from collections import deque
import timeit
from datetime import datetime

import numpy as np


class TimeMonitor(object):
    def __init__(self, name):
        self.name = name
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

    def get_state(self):
        return {'epoch_durations': self.epoch_durations}

    def set_state(self, states):
        self.epoch_durations = states['epoch_durations']


class BatchHistoryMonitor(object):
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

    def __init__(self, name, weighted: bool = False):
        self.name = name
        self.is_weighted = weighted

        # State:
        self.current_epoch_batch_values = []
        self.current_epoch_batch_weights = []
        self.average_per_epoch = []
        self.current_epoch = -1
        self.ever_min = np.inf
        self.ever_max = -np.inf

    def update(self, value, weight=1):
        """
        Adds a batch value to the epoch.
        Note. Does not save the update if value is inf.

        Parameters
        ----------
        value: The value to be monitored.
        weight: The weight in the average. For instance, for a loss monitor,
            you should measure the loss average.
        """
        if np.isinf(value):
            self.ever_max = value
            return

        if value > self.ever_max:
            self.ever_max = value
        if value < self.ever_min:
            self.ever_min = value

        self.current_epoch_batch_values.append(value)

        if self.is_weighted:
            self.current_epoch_batch_weights.append(weight)

    def start_new_epoch(self):
        assert len(self.average_per_epoch) == self.current_epoch + 1, \
            "Did you forget to end previous epoch? Number of epoch values " \
            "is {} but monitor's current epoch is {} (i.e. the {}th)" \
            .format(len(self.average_per_epoch), self.current_epoch,
                    self.current_epoch + 1)

        self.current_epoch += 1
        self.current_epoch_batch_values = []
        self.current_epoch_batch_weights = []

    def end_epoch(self):
        """
        Compute mean of current epoch and add it to means values.
        """
        if len(self.current_epoch_batch_values) == 0:
            logging.info(
                "No batch in monitor, or was inf for all batches in this "
                "epoch. Please supervise.")
            # All batches were inf and thus not stored. Or no batch.
            self.average_per_epoch.append(np.nan)
            return

        if not self.is_weighted:
            mean_value = np.mean(self.current_epoch_batch_values)
        else:
            mean_value = sum(np.multiply(self.current_epoch_batch_values,
                                         self.current_epoch_batch_weights))
            mean_value /= sum(self.current_epoch_batch_weights)

        self.average_per_epoch.append(mean_value)

    def get_state(self):
        # Not saving current batch values. Checkpoints should be saved only at
        # the end of epochs.
        return {'average_per_epoch': self.average_per_epoch,
                'current_epoch': self.current_epoch,
                'ever_max': self.ever_max,
                'ever_min': self.ever_min
                }

    def set_state(self, state):
        self.average_per_epoch = state['average_per_epoch']
        self.current_epoch = state['current_epoch']
        self.ever_max = state['ever_max']
        self.ever_min = state['ever_min']


class BestEpochMonitor(object):
    """
    Object to stop training early if the loss doesn't improve after a given
    number of epochs ("patience").
    """

    def __init__(self, name, patience: int, patience_delta: float = 1e-6):
        """
        Parameters
        -----------
        patience: int
            Maximal number of bad epochs we allow.
        patience_delta: float, optional
            Precision term to define what we consider as "improving": when the
            loss is at least min_eps smaller than the previous best loss.
        """
        self.name = name
        self.patience = patience
        self.min_eps = patience_delta

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
