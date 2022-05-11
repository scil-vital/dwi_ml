# -*- coding: utf-8 -*-
import collections
import logging
from tqdm import tqdm


class TqdmLoggingHandler(logging.StreamHandler):
    """Avoid tqdm progress bar interruption by logger's output to console"""
    # see logging.StreamHandler.eval method:
    # https://github.com/python/cpython/blob/d2e2534751fd675c4d5d3adc208bf4fc984da7bf/Lib/logging/__init__.py#L1082-L1091
    # and tqdm.write method:
    # https://github.com/tqdm/tqdm/blob/f86104a1f30c38e6f80bfd8fb16d5fcde1e7749f/tqdm/std.py#L614-L620
    # And solution here:
    # https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, end=self.terminator)
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def format_dict_to_str(d, indent=1):
    indentation = indent * "    "
    return ("\n" + indentation) + ("\n" + indentation).join(
        "{!r}: {},".format(k, _format_val_to_str(v, indent+1))
        for k, v in d.items())


def _format_val_to_str(v, indent):
    if isinstance(v, collections.Mapping):
        return format_dict_to_str(v, indent)
    else:
        return v


def make_logger_tqdm_fitted(logger):
    """Possibility to use a tqdm-compatible logger in case the model
    is used through a tqdm progress bar."""
    if len(logger.handlers) == 0:
        logger.addHandler(TqdmLoggingHandler())
        logger.propagate = False


def make_logger_normal(logger):
    # toDo
    #  I've tried a lot of things. self.logger.remove_handlers,
    #  self.logger.handlers = [] and more. Not working.
    pass
