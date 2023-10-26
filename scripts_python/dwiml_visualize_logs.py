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
    parser = argparse.ArgumentParser(description=__doc__,
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
    parser.add_argument("--save_to_csv", metavar='my_file.csv',
                        help="If set, save the resulting logs as a csv file.")
    parser.add_argument('--xlim', type=int,
                        help="Graph's xlim. Makes little sense with more than "
                             "one graph. Format: max_epoch ")
    parser.add_argument('--ylims', type=float, nargs=2,
                        help="Graph's ylim. Makes little sense with more than "
                             "one graph. Format: ymin ymax ")

    add_overwrite_arg(parser)
    return parser


def visualize_logs(logs: Dict[str, Dict[str, np.ndarray]], graphs, nb_rows,
                   writer=None, xlim=None, ylims=None):
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
                        axs[i].plot(logs[exp_name][key], linestyle=style,
                                    label=legend, color=color_val)

                        if writer is not None:
                            writer.writerow([exp_name, key] +
                                            list(logs[exp_name][key]))

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
                           writer=writer, xlim=args.xlim, ylims=args.ylims)
    else:
        visualize_logs(loaded_logs, graphs, args.nb_plots_per_fig,
                       xlim=args.xlim, ylims=args.ylims)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
