#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import pathlib
from argparse import RawTextHelpFormatter
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def visualize_logs(logs: Dict[str, np.ndarray]):
    if len(logs) > 1:
        fig, ax = plt.subplots(nrows=len(logs))

        for i, (log_name, data) in enumerate(logs.items()):
            ax[i].set_title(log_name)
            ax[i].plot(data)
    else:
        key, data = logs.popitem()
        plt.title(key)
        plt.plot(data)

    plt.tight_layout()
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("path", type=str, help="Path to the experiment folder")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)

    path = args.path

    if not pathlib.Path(path).exists():
        raise ValueError("Experiment folder does not exist: {}".format(path))

    logs_path = pathlib.Path(path, 'logs')
    if not logs_path.exists():
        raise ValueError("No log folder exists!")

    log_files = list(logs_path.glob('*.npy'))
    log_files_str = [str(f) for f in log_files]

    logging.info("Found files: {}".format(log_files_str))
    logs = {}
    for log_file in log_files:
        log_name = log_file.stem
        data = np.load(log_file)
        logs[log_name] = data

    visualize_logs(logs)


if __name__ == '__main__':
    main()
