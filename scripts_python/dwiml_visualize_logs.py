#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import pathlib
from argparse import RawTextHelpFormatter
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

MAX_PLOT = 3


def visualize_logs(logs: Dict[str, np.ndarray]):
    nb_plots_left = len(logs)
    all_keys = list(logs.keys())
    key_idx = 0
    while nb_plots_left > 0:
        next_nb_plots = min(nb_plots_left, MAX_PLOT)
        fig, axs = plt.subplots(nrows=next_nb_plots)
        if next_nb_plots == 1:
            axs = [axs]

        for i in range(0, next_nb_plots):
            key = all_keys[key_idx]
            print("{}".format(key))
            axs[i].set_title(key)
            axs[i].plot(logs[key])
            key_idx += 1

        nb_plots_left -= next_nb_plots


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
    logging.getLogger().setLevel(level=logging.INFO)

    path = args.path

    if not pathlib.Path(path).exists():
        raise ValueError("Experiment folder does not exist: {}".format(path))

    logs_path = pathlib.Path(path, 'logs')
    if not logs_path.exists():
        raise ValueError("No log folder exists!")

    log_files = list(logs_path.glob('*.npy'))
    log_files_str = [str(f) for f in log_files]

    logs = {}
    for log_file in log_files:
        log_name = log_file.stem
        data = np.load(log_file)
        logs[log_name] = data

    visualize_logs(logs)


if __name__ == '__main__':
    main()
