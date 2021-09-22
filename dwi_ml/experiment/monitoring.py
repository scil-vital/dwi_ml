# -*- coding: utf-8 -*-
from collections import deque
import timeit

import numpy as np


class ValueHistoryMonitor(object):
    """ History of some value for each iteration during training, and mean
    value for each epoch.

    Example of usage: History of the loss during training.
        loss_monitor = ValueHistoryMonitor()
        ...
        # Call update at each iteration
        loss_monitor.update(2.3)
        ...
        loss_monitor.avg  # returns the average loss
        ...
        loss_monitor.end_epoch()  # call at epoch end
        ...
        loss_monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self, name):
        self.name = name
        self.all_updates_history = []  # The list of values
        self.epochs_means_history = []  # The list of averaged values per epoch
        self.current_epoch_history = []  # The list of values in current epoch

    @property
    def nb_updates(self):
        return len(self.all_updates_history)

    @property
    def nb_epochs(self):
        return len(self.epochs_means_history)

    def update(self, value):
        """
        Note. Does not save the update if value is inf.

        Parameters
        ----------
        value: The value of the loss
        """
        if np.isinf(value):
            return

        self.all_updates_history.append(value)
        self.current_epoch_history.append(value)

    def end_epoch(self):
        """
        Reset the current epoch.
        Append the average of this epoch's losses.
        """
        self.epochs_means_history.append(np.mean(self.current_epoch_history))
        self.current_epoch_history = []

    def get_state(self):
        return {'name': self.name,
                'all_updates_history': self.all_updates_history,
                'current_epoch_history': self.current_epoch_history,
                'epochs_means_history': self.epochs_means_history}

    def set_state(self, state):
        self.name = state['name']
        self.all_updates_history = state['all_updates_history']
        self.current_epoch_history = state['current_epoch_history']
        self.epochs_means_history = state['epochs_means_history']


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
        """
        if self.best_value is None:
            # First epoch. Setting values.
            self.best_value = loss
            self.best_epoch = epoch
            self.n_bad_epochs = 0
        elif loss < self.best_value - self.min_eps:
            # Improving from at least eps.
            self.best_value = loss
            self.best_epoch = epoch
            self.n_bad_epochs = 0
        else:
            # Not improving enough
            self.n_bad_epochs += 1

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
