import timeit
from collections import deque

import numpy as np

"""
Classes in this file:
    ValueMonitor
    EarlyStopping
    IterTimer

Functions in this file:
    compute_gradient_norm
"""


class LossHistoryMonitor(object):
    """ History of the loss during training. (Lighter version of MetricHistory)
    Usage:
        monitor = LossHistory()
        ...
        # Call update at each iteration
        monitor.update(2.3)
        ...
        monitor.avg  # returns the average loss
        ...
        monitor.end_epoch()  # call at epoch end
        ...
        monitor.epochs  # returns the loss curve as a list
    """

    def __init__(self, name):
        self.name = name
        self.history = []
        self.epochs = []
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_iter = 0
        self.num_epochs = 0

    def update(self, value):
        if np.isinf(value):
            return

        self.history.append(value)
        self.sum += value
        self.count += 1
        self._avg = self.sum / self.count
        self.num_iter += 1

    @property
    def avg(self):
        return self._avg

    def end_epoch(self):
        self.epochs.append(self._avg)
        self.sum = 0.0
        self.count = 0
        self._avg = 0.0
        self.num_epochs += 1

    def get_state(self):
        return {'name': self.name,
                'history': self.history,
                'epochs': self.epochs,
                'num_iter': self.num_iter,
                'num_epochs': self.num_epochs}

    def set_state(self, state):
        self.name = state['name']
        self.history = state['history']
        self.epochs = state['epochs']
        self.num_iter = state['num_iter']
        self.num_epochs = state['num_epochs']


class EarlyStopping(object):
    """ Object to stop training early if the loss doesn't improve after a given
    number of epochs. """

    def __init__(self, patience: int, min_eps: float = 1e-6):
        self.patience = patience
        self.min_eps = min_eps

        self.best = None
        self.n_bad_epochs = 0

    def step(self, loss):
        """ Manage a loss step. Returns True if training should stop.

        Parameters
        ----------
        loss : float
            Loss value for a new training epoch

        Returns
        -------
        True if the number of epochs without improvements is more than the
        patience.
        """
        if self.best is None:
            self.best = loss
            return False

        if loss < self.best - self.min_eps:
            self.best = loss
            self.n_bad_epochs = 0
        else:
            self.n_bad_epochs += 1

        if self.n_bad_epochs >= self.patience:
            return True

        return False

    def get_state(self):
        """ Get object state """
        return {'patience': self.patience,
                'min_eps': self.min_eps,
                'best': self.best,
                'n_bad_epochs': self.n_bad_epochs}

    def set_state(self, state):
        """ Set object state """
        self.patience = state['patience']
        self.min_eps = state['min_eps']
        self.best = state['best']
        self.n_bad_epochs = state['n_bad_epochs']


class IterTimer(object):
    """
    Description to do
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


def compute_gradient_norm(parameters):
    """Compute the gradient norm of the provided iterable parameters.
    (Machine learning gradient descent, not dwi gradients!)

    Parameters
    ----------
    parameters : list of torch.Tensor
        Model parameters after loss.backwards() has been called

    Returns
    -------
    total_norm : float
        The total gradient norm of the parameters
    """
    total_norm = 0.
    for p in parameters:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 3)
    return total_norm
