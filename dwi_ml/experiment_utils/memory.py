# -*- coding: utf-8 -*-
import torch
import pynvml

BYTES_IN_GB = 1024 ** 3


def log_gpu_memory_usage(logger=None):
    if torch.cuda.is_available():
        # From torch
        total_free, total = torch.cuda.mem_get_info()
        # total can also be found with:
        # torch.cuda.get_device_properties(0).total_memory

        allocated = torch.cuda.memory_allocated() / BYTES_IN_GB
        cached = torch.cuda.memory_reserved() / BYTES_IN_GB

        # From whole system
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(h)

        # (Peak since beginning of program:
        # ALLOCATED {:>6.3f}GB; CACHED: {:>6.3f}GB)
        # torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        # torch.cuda.max_memory_reserved() / BYTES_IN_GB,
        msg = ("GPU: TOTAL: Asking torch: {:>6.3f}GB. Asking pynvml: {:>6.3f}GB\n"
               "   - Used asking torch: ALLOCATED: {:>6.3f}GB + CACHED: {:>6.3f}GB"
               " = {:>6.3f}GB\n"
               "   - Total used including other processes: Asking pynvml: {:>6.3f}GB.\n"
               "   - Free: Asking torch: {:>6.3f}GB. Asking pynvml: {:>6.3f}GB\n"
               .format(total / BYTES_IN_GB, info.total / BYTES_IN_GB,
                       allocated, cached, allocated + cached,
                       info.used / BYTES_IN_GB,
                       total_free / BYTES_IN_GB, info.free / BYTES_IN_GB))
        if logger is None:
            print(msg)
        else:
            logger.debug(msg)
