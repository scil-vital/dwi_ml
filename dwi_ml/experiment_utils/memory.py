# -*- coding: utf-8 -*-
import logging

import torch

BYTES_IN_GB = 1024 ** 3


def log_gpu_memory_usage(logger=None):
    # Note: torch.cuda.memory_cached has been renamed to
    # torch.cuda.memory_reserved in newest versions
    if torch.cuda.is_available():
        total_used, total = torch.cuda.mem_get_info()

        msg = ("GPU: TOTAL: {:>6.3f}GB.\n"
               "   Current values: ALLOCATED: {:>6.3f}GB; CACHED: {:>6.3f}GB\n"
               "   Total used including other processes: {:>6.3f}GB\n"
               "   Peak since beginning of program: ALLOCATED {:>6.3f}GB); "
               "CACHED: {:>6.3f}GB"
               .format(total / BYTES_IN_GB,
                       torch.cuda.memory_allocated() / BYTES_IN_GB,
                       torch.cuda.memory_reserved() / BYTES_IN_GB,
                       torch.cuda.max_memory_allocated() / BYTES_IN_GB,
                       torch.cuda.max_memory_reserved() / BYTES_IN_GB,
                       total_used / BYTES_IN_GB))
        if logger is None:
            print(msg)
        else:
            logger.debug(msg)
