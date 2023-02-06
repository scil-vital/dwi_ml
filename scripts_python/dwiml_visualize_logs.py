#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import logging
import os
import pathlib
from argparse import RawTextHelpFormatter
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

log_styles = ['-', '--', '_.', ':']
nb_styles = 4
exp_colors = ['b', 'r', 'g', 'k', 'o']
nb_colors = 5


def parse_args():
    parser = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("paths", type=str, nargs='+',
                        help="Path to the experiment folder (s). If more than "
                             "one, they are shown superposed.")
    parser.add_argument("--graph", action='append', nargs='+',
                        help="Name of the logs to add to one graph. Can be "
                             "used many times. \n"
                             "Ex: --graph log1 --graph log2 log3\n"
                             "If not set, all logs are shown separately.")
    parser.add_argument("--nb_plots_per_fig", type=int, default=3,
                        help="Number of (rows) of plot per figure.")

    args = parser.parse_args()
    return args


def visualize_logs(logs: List[Dict[str, np.ndarray]], graphs, nb_rows):
    nb_exp = len(logs)

    nb_plots_left = len(graphs)
    current_graph = -1
    while nb_plots_left > 0:
        next_nb_plots = min(nb_plots_left, nb_rows)
        fig, axs = plt.subplots(nrows=next_nb_plots)
        if next_nb_plots == 1:
            axs = [axs]

        # For each subplot in this figure
        for i in range(next_nb_plots):
            current_graph += 1
            keys = graphs[current_graph]
            print(keys)

            # For each log to show in that plot
            for j in range(len(keys)):
                style = log_styles[j % nb_styles]
                key = keys[j]
                axs[i].set_title(key)

                # For each experiment to show:
                for exp in range(nb_exp):
                    color = exp_colors[exp % nb_colors]
                    if key in logs[exp]:
                        axs[i].plot(logs[exp][key], linestyle=style,
                                    color=color,
                                    label='exp{}, {}'.format(exp, key))

                axs[i].legend()

        nb_plots_left -= next_nb_plots

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    logging.getLogger().setLevel(level=logging.INFO)

    # One element per experiment
    loaded_logs = []  # List of dict
    user_required_names = set(itertools.chain(*args.graph)
                              ) if args.graph is not None else None
    graphs = args.graph or set()

    # Loading all.
    for path in args.paths:
        if not pathlib.Path(path).exists():
            raise ValueError("Experiment folder does not exist: {}"
                             .format(path))

        log_path = pathlib.Path(path, 'logs')
        if not log_path.exists():
            raise ValueError("No log folder exists for experiment {}!"
                             .format(path))

        if args.graph is None:
            # One graph per log.
            files_to_load = list(log_path.glob('*.npy'))
        else:
            files_to_load = []
            for n in user_required_names:
                nn = pathlib.Path(log_path, n + '.npy')
                if os.path.isfile(nn):
                    files_to_load.append(nn)
                else:
                    print("File {} not found in path {}. Skipping."
                          .format(n, log_path))
        print(files_to_load)
        names_to_load = [n.stem for n in files_to_load]

        exp_logs = {}
        for i in range(len(files_to_load)):
            data = np.load(files_to_load[i])
            exp_logs.update({names_to_load[i]: data})
        loaded_logs.append(exp_logs)

        if args.graph is None:
            graphs |= set(names_to_load)

    if args.graph is None:
        graphs = list(graphs)
        graphs = [[g] for g in graphs]

    visualize_logs(loaded_logs, graphs, args.nb_plots_per_fig)


if __name__ == '__main__':
    main()
