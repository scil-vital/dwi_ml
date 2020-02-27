import numpy as np


class LossHistoryMonitor(object):
    """ History of the loss during training. (Lighter version of MetricHistory)
    Usage:
        monitor = LossHistoryMonitor()
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
        self.history = []  # The list of loss values
        self.epochs = []  # The list of averaged loss values per epoch
        self.current_history = [] # The list of loss values in current epoch

    @property
    def total_iter_count(self):
        return len(self.history)

    @property
    def num_epochs(self):
        return len(self.epochs)

    def update(self, value):
        """
        Note. Does not save the update if value is inf.

        Parameters
        ----------
        value: The value of the loss
        """
        if np.isinf(value):
            return

        self.history.append(value)
        self.current_history.append(value)

    def end_epoch(self):
        """
        Reset the current epoch.
        Append the average of this epoch's losses.
        """
        self.epochs.append(np.mean(self.current_history))
        self.current_history = []

    def get_state(self):
        return {'name': self.name,
                'history': self.history,
                'epochs': self.epochs}

    def set_state(self, state):
        self.name = state['name']
        self.history = state['history']
        self.epochs = state['epochs']


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

