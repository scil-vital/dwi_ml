#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import RawTextHelpFormatter
import csv
import itertools
import logging
import os
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import numpy as np

from scilpy.io.utils import assert_outputs_exist, add_overwrite_arg

log_styles = ['-', '--', '_.', ':']
nb_styles = 4


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=RawTextHelpFormatter)
    p.add_argument("paths", type=str, nargs='+',
                   help="Path to the experiment folder (s). If more than "
                        "one, they are shown superposed.")
    p.add_argument("--graph", action='append', nargs='+',
                   help="Name of the logs to add to one graph. Can be "
                        "used many times. \n"
                        "Ex: --graph log1 --graph log2 log3\n"
                        "If not set, all logs are shown separately.")
    p.add_argument("--nb_plots_per_fig", type=int, default=3,
                   help="Number of (rows) of plot per figure.")
    p.add_argument("--save_to_csv", metavar='my_file.csv',
                   help="If set, save the resulting logs as a csv file \n"
                        "(chosen --graph only, if any).")
    p.add_argument('--xlim', type=int, metavar='epoch_max',
                   help="All graphs' xlim.")
    p.add_argument('--ylims', type=float, nargs=2, metavar='ymin ymax',
                   help="All graph's ylim. (Makes little sense with more "
                        "than one graph.)")
    p.add_argument('--remove_outliers', action='store_true',
                   help="If set, remove outliers (3 std) from plot line and "
                        "adds a star sign instead.\n"
                        "Outliers are not removed in the CSV file, if any.")

    add_overwrite_arg(p)
    return p


def plot_one_line(ax, data, color, style, exp_name, legend, remove_outliers):
    epochs = np.arange(0, len(data))
    if remove_outliers:
        std3 = 3 * np.std(data)
        mean = np.mean(data)
        outlier_idx = np.logical_or(data < mean - std3, data > mean + std3)

        print("{}: Found {} outliers out of [{}, {}]."
              .format(exp_name, outlier_idx.sum(),
                      mean - std3, mean + std3))
        ax.scatter(epochs[outlier_idx], data[outlier_idx],
                   marker='*', color=color)
        epochs = epochs[~outlier_idx]
        data = data[~outlier_idx]

    ax.plot(epochs, data, linestyle=style, label=legend, color=color)


def visualize_logs(logs: Dict[str, Dict[str, np.ndarray]], graphs, nb_rows,
                   writer=None, xlim=None, ylims=None, remove_outliers=False):
    exp_names = list(logs.keys())
    writer.writerow(['Experiment name', "Log name", "Epochs..."])

    jet = plt.get_cmap('jet')
    c_norm = colors.Normalize(vmin=0, vmax=len(exp_names))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

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
            print("Next graph: ", keys)

            # For each log to show in that plot
            for j in range(len(keys)):
                style = log_styles[j % nb_styles]
                key = keys[j]
                axs[i].set_title(key)

                # For each experiment to show:
                for exp, exp_name in enumerate(exp_names):
                    color_val = scalar_map.to_rgba(exp)
                    legend = exp_name
                    if len(keys) > 1:
                        legend += ', ' + key
                    if key in logs[exp_name]:
                        data = logs[exp_name][key]
                        plot_one_line(axs[i], data, color_val, style, exp_name,
                                      legend, remove_outliers)

                        if writer is not None:
                            writer.writerow([exp_name, key] + list(data))

                axs[i].legend()
                if xlim is not None:
                    axs[i].set_xlim([0, xlim])
                if ylims is not None:
                    axs[i].set_ylim(ylims)

        nb_plots_left -= next_nb_plots


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.getLogger().setLevel(level=logging.INFO)

    assert_outputs_exist(parser, args, args.save_to_csv)

    # One element per experiment
    loaded_logs = {}  # exp: dict of logs
    user_required_names = set(itertools.chain(*args.graph)
                              ) if args.graph is not None else None
    graphs = args.graph or set()

    # Loading all.
    for path in args.paths:
        if not pathlib.Path(path).exists():
            raise ValueError("Experiment folder does not exist: {}"
                             .format(path))

        exp_name = os.path.basename(os.path.normpath(path))
        print("Loading experiment {} from path {}".format(exp_name, path))

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
                    print("File {}.npy not found in path {}. Skipping."
                          .format(n, log_path))
        names_to_load = [n.stem for n in files_to_load]

        exp_logs = {}
        for i in range(len(files_to_load)):
            data = np.load(files_to_load[i])
            exp_logs.update({names_to_load[i]: data})
        loaded_logs[exp_name] = exp_logs

        if args.graph is None:
            graphs |= set(names_to_load)

    if args.graph is None:
        graphs = list(graphs)
        graphs = [[g] for g in graphs]

    if args.save_to_csv:
        print("Will save results in file {}".format(args.save_to_csv))
        with open(args.save_to_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            visualize_logs(loaded_logs, graphs, args.nb_plots_per_fig,
                           writer=writer, xlim=args.xlim, ylims=args.ylims,
                           remove_outliers=args.remove_outliers)
    else:
        visualize_logs(loaded_logs, graphs, args.nb_plots_per_fig,
                       xlim=args.xlim, ylims=args.ylims,
                       remove_outliers=args.remove_outliers)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
