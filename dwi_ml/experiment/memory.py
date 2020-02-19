import logging

import torch

"""
Functions in this file:
    log_gpu_memory_usage
"""

BYTES_IN_GB = 1024 ** 3


def log_gpu_memory_usage():
    logging.debug("GPU: "
                  "ALLOCATED: {:>6.3f}GB (max: {:>6.3f}GB);  "
                  "CACHED: {:>6.3f}GB (max: {:>6.3f}GB)"
                  "".format(torch.cuda.memory_allocated() / BYTES_IN_GB,
                            torch.cuda.max_memory_allocated() / BYTES_IN_GB,
                            torch.cuda.memory_cached() / BYTES_IN_GB,
                            torch.cuda.max_memory_cached() / BYTES_IN_GB))
