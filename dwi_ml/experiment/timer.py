import sys
import time
from time import time

"""
A timer class printing elapsed time in color.
(Could it be useful in vitalabai? Would others use it??)
"""


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


class Timer:
    """
    Times code within a `with` statement, optionally adding color.
    """

    def __init__(self, txt, newline=False, color=None):
        try:
            prepend = (COLOR_CODES[color] if color else '')
            append = (COLOR_CODES['reset'] if color else '')
        except KeyError:
            prepend = ''
            append = ''

        self.txt = prepend + txt + append
        self.newline = newline

    def __enter__(self):
        self.start = time()
        if not self.newline:
            print(self.txt + "... ", end="")
            sys.stdout.flush()
        else:
            print(self.txt + "... ")

    def __exit__(self, type, value, tb):
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
