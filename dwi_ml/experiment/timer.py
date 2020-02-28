# -*- coding: utf-8 -*-
import sys
import time
from time import time
import timeit
from collections import deque

import numpy as np


COLOR_CODES = {
    'black': '\u001b[30m',
    'red': '\u001b[31m',
    'green': '\u001b[32m',
    'yellow': '\u001b[33m',
    'blue': '\u001b[34m',
    'magenta': '\u001b[35m',
    'cyan': '\u001b[36m',
    'white': '\u001b[37m',
    'reset': '\u001b[0m'
}


# Checked!
class Timer:
    """
    Example of usage:
    with Timer("TimedFunction", newline=True, color='blue'):
        ... # do something

    """
    def __init__(self, txt: str, newline: bool = False, color: str = None):
        """
        Parameters
        ----------
        txt: str
            Name of this timer.
        newline: bool
            Wheter you want prints to end with newlines.
        color: str
            One of 'black', 'red', 'green', 'yellow', 'blue', 'magenta',
            'cyan', or 'white'.
        """
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append
        self.newline = newline

    def __enter__(self):
        """
        Used at the beginning of the section inside "With Timer()".
        Prints txt and starts time.
        """
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
        """
        Used at the end of the section inside "With Timer()".
        Prints 'done' and the final time.
        """
        if self.newline:
            print(self.txt + " done in ", end="")

        print("{:.2f} sec.".format(time() - self.start))


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
