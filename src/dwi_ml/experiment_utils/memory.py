# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import torch
import pynvml
import gc

BYTES_IN_GB = 1024 ** 3


def torch_reset_peaks_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def log_gpu_general_info(logger_info=None, context: str = None):
    """
    Prints general info. Available GPU, used by other processes, etc.
    Intended to be used at the start of the experiment.
    (other logs below are on logger.debug)

    Params
    ------
    logger_info: logging.Logger
        If set, will show on logger.info. Else, will print.
    context: str
        IF set, will add additional information in the message.
    """
    if torch.cuda.is_available():
        # From torch
        total_free, total = torch.cuda.mem_get_info()
        # total can also be found with:
        # torch.cuda.get_device_properties(0).total_memory

        # From whole system
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)

        # (Peak since beginning of program:
        # ALLOCATED {:>6.3f}GB; CACHED: {:>6.3f}GB)
        # torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        # torch.cuda.max_memory_reserved() / BYTES_IN_GB,
        msg = ("GPU: {}\n"
               "   - TOTAL: Asking torch: {:>6.3f}GB. Asking pynvml: {:>6.3f}GB\n"
               "   - CURRENTLY USED:\n"
               "   -    Used (including other processes): {:>6.3f}GB.\n"
               "   -    Free: Asking torch: {:>6.3f}GB. Asking pynvml: {:>6.3f}GB\n"
               .format(context,
                       total / BYTES_IN_GB, info.total / BYTES_IN_GB,
                       info.used / BYTES_IN_GB,
                       total_free / BYTES_IN_GB, info.free / BYTES_IN_GB))
        if logger_info is None:
            print(msg)
        else:
            logger_info.info(msg)


def log_currently_allocated(logger_debug=None, context: str = None):
    """
    Prints currently allocated GPU memory.

    Params
    ------
    logger_debug: logging.Logger
        If set, will show on logger.info. Else, will print.
    context: str
        IF set, will add additional information in the message.
    """
    if torch.cuda.is_available():
        # From torch
        # total can also be found with:
        # torch.cuda.get_device_properties(0).total_memory

        allocated = torch.cuda.memory_allocated() / BYTES_IN_GB
        cached = torch.cuda.memory_reserved() / BYTES_IN_GB

        # (Peak since beginning of program:
        # ALLOCATED {:>6.3f}GB; CACHED: {:>6.3f}GB)
        # torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        # torch.cuda.max_memory_reserved() / BYTES_IN_GB,
        msg = ("GPU: {}\n"
               "  - allocated: {:>6.3f}GB + cached: {:>6.3f}GB = {:>6.3f}GB"
               .format(context, allocated, cached, allocated + cached))
        if logger_debug is None:
            print(msg)
        else:
            logger_debug.debug(msg)

def log_max_allocated(logger_debug=None, context: str = None):
    if torch.cuda.is_available():
        msg = ("GPU: {}\n"
               "  - max memory: {:>6.3f}GB"
               .format(context,
                       torch.cuda.max_memory_allocated() / BYTES_IN_GB))
        if logger_debug is None:
            print(msg)
        else:
            logger_debug.debug(msg)

def log_gpu_per_tensor(logger_debug=None, context: str = None):
    """
    Prints currently alive Tensors and Variables
    This is extensive and intended for debugging.
    When fails, tends to keep data in memory. Use
       sudo fuser -v /dev/nvidia*
       sudo kill -9 pid
    """
    all_tensors_gpu = [[],[]]  # Name, size (nb elements),
    all_tensors_cpu = [[],[]]  # Name, size (nb elements),

    ignored = 0

    for obj in gc.get_objects():
        _tensor = None
        try:
            if torch.is_tensor(obj) or (
                    hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                _tensor = obj
            else:
                raise TypeError()
        except:
            ignored += 1

        # Recording tensor info
        if _tensor is not None:
            if obj.is_cuda:
                # GPU
                all_tensors_gpu[0].append(type(obj).__name__)
                size_GB = obj.nelement() * obj.element_size() / BYTES_IN_GB
                all_tensors_gpu[1].append(size_GB)
            else:
                # CPU
                all_tensors_cpu[0].append(type(obj).__name__)
                size_GB = obj.nelement() * obj.element_size() / BYTES_IN_GB
                all_tensors_cpu[1].append(size_GB)

    # ========== Ok.==============
    msg = ("Number of objects found: {}\n"
           "    - Total : {} tensor obj ({} other types ignored), {} GPU, {} CPU"
          .format(context,
                  len(all_tensors_gpu[0]) + len(all_tensors_cpu[0]), ignored,
                  len(all_tensors_gpu[0]), len(all_tensors_cpu[0])))

    counts = Counter(all_tensors_gpu[0])
    keys = list(counts.keys())
    keys = sorted(keys)
    for key in keys:
        sizes = [s for (n, s) in
                 zip(all_tensors_gpu[0], all_tensors_gpu[1]) if n == key ]
        msg += ("\n     - GPU: Found {} x {}. Total size: {:.2f}. Biggest size: {:.2f}"
              .format(counts[key], key, np.sum(sizes), np.max(sizes)))

    if logger_debug is None:
        print(msg)
    else:
        logger_debug.debug(msg)


def plot_gpu_usage():
    # See here https://huggingface.co/blog/train_memory
    raise NotImplementedError
