# -*- coding: utf-8 -*-
import logging

import torch

BYTES_IN_GB = 1024 ** 3


def log_gpu_memory_usage(logger=None):
    # Note: torch.cuda.memory_cached has been renamed to
    # torch.cuda.memory_reserved in newest versions
    msg = ("GPU: ALLOCATED: {:>6.3f}GB (max: {:>6.3f}GB);  "
           "CACHED: {:>6.3f}GB (max: {:>6.3f}GB)"
           .format(torch.cuda.memory_allocated() / BYTES_IN_GB,
                   torch.cuda.max_memory_allocated() / BYTES_IN_GB,
                   torch.cuda.memory_reserved() / BYTES_IN_GB,
                   torch.cuda.max_memory_reserved() / BYTES_IN_GB))
    if logger is None:
        logging.debug(msg)
    else:
        logger.debug(msg)
