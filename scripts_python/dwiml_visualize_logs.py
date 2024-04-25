#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
from argparse import RawTextHelpFormatter
import csv
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from scilpy.io.utils import assert_outputs_exist, add_overwrite_arg

from dwi_ml.io_utils import add_verbose_arg
from dwi_ml.viz.logs_plots import visualize_logs


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=RawTextHelpFormatter)
    p.add_argument("experiments", type=str, nargs='+',
                   help="Path to the experiment folder (s). If more than "
                        "one, they are shown superposed.")
    p.add_argument("--graph", action='append', nargs='+', dest='graphs',
                   help="By default, all logs are shown separately. Instead, "
                        "you may specify the \ngraphs to sow. Name of the "
                        "logs to add to one graph. This option can be used "
                        "many times. \n"
                        "Ex: >> --graph train_loss_monitor_per_epoch\n"
                        "Ex: >> --graph log1 --graph log2 log3\n"
                        "Optionally, you can add the 2-valued ylims for each "
                        "graph. These value will \nsupersede the given "
                        "--ylim, if any.\nEx: >> --graph log1 0 100\n")
    p.add_argument("--nb_plots_per_fig", type=int, default=3, metavar='n',
                   help="Number of (rows) of plot per figure. Default: 3.")
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
    p.add_argument('--allow_missing', action='store_true',
                   help="If true, ignore experiments with missing logs.")

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def _parse_graphs_arg(parser, args):
    """Parse args.graphs"""
    if args.graphs is None:
        return None, None
    else:
        graphs = []
        graphs_ylims = []
        for graph in args.graphs:
            # Verify if gave the ylims.
            try:
                min_y = float(graph[-2])
                max_y = float(graph[-1])
                graphs.append(graph[:-2])
                graphs_ylims.append([min_y, max_y])
            except ValueError:
                try:
                    _ = float(graph[-1])
                    parser.error("For option --graph {}: you seem to have "
                                 "added a single float value at the end. "
                                 "Ylim requires two values.".format(graph))
                except ValueError:
                    # Ok.
                    graphs.append(graph)
                    graphs_ylims.append(None)

            # Add .npy to log names if not added by user.
            for i, log in enumerate(graphs[-1]):
                f, ext = os.path.splitext(log)
                if ext == '':
                    graphs[-1][i] = log + '.npy'
                else:
                    if ext != '.npy':
                        parser.error("The values to the --graph option "
                                     "should be log names, but got {} (try "
                                     "removing the extension).".format(log))

        return graphs, graphs_ylims


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(level=args.verbose)

    # Do not set DEBUG level for matplotlib
    plt.set_loglevel('WARNING')
    pil_logger = logging.getLogger('PIL')
    pil_logger.setLevel('WARNING')

    # Verifications
    assert_outputs_exist(parser, args, [], args.save_to_csv)
    graphs, graphs_ylims = _parse_graphs_arg(parser, args)

    # Loading all.
    loaded_logs = {}  # dict of logs
    for exp_path in args.experiments:
        # Verifications
        if not pathlib.Path(exp_path).exists():
            raise ValueError("Experiment folder does not exist: {}"
                             .format(exp_path))
        exp_name = os.path.basename(os.path.normpath(exp_path))
        logs_path = pathlib.Path(exp_path, 'logs')

        if not logs_path.exists():
            raise ValueError("Logs folder does not exist for experiment {}!"
                             .format(exp_path))

        # Find the list of all log files to load for this experiment.
        if args.graphs is None:
            # One graph per log.
            files_to_load = list(logs_path.glob('*.npy'))
            all_logs = [f.stem for f in files_to_load]
            if graphs is None:
                # Using the first experiment to set the list of expected logs.
                graphs = all_logs
            else:
                if args.allow_missing:
                    # Simply use all log names from all exp.
                    graphs = list(set(graphs.extend(all_logs)))
                else:
                    # Compare with first experiment's log
                    if set(graphs) != set(all_logs):
                        parser.error(
                            "The experiments do not all have the same logs. "
                            "To allow running, use --allow_missing.")
        else:
            # Use options --graph.
            files_to_load = []
            for graph in graphs:
                for required_log in graph:
                    log_path = pathlib.Path(logs_path, required_log)
                    if os.path.isfile(log_path):
                        if log_path not in files_to_load:
                            files_to_load.append(log_path)
                    else:
                        if not args.allow_missing:
                            parser.error("File {} not found in path {}. "
                                         "Skipping.".format(graph, logs_path))

        # Loading
        logging.info("Loading logs from experiment {}.".format(exp_name))
        logging.debug("   (logs path: {})".format(logs_path))

        exp_logs_dict = {}
        for file in files_to_load:
            log_name = os.path.basename(file)
            data = np.load(file)
            exp_logs_dict.update({log_name: data})
        loaded_logs[exp_name] = exp_logs_dict

    if args.graphs is None:
        # Formatting the final graphs choice.
        graphs = [[log] for log in graphs]
        graphs_ylims = [None for g in graphs]

    # Final formatting of ylims.
    if args.ylims:
        graphs_ylims = [args.ylims if ylim is None else ylim
                        for ylim in graphs_ylims]

    if args.save_to_csv:
        logging.info("Will save results in file {}".format(args.save_to_csv))
        with open(args.save_to_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            visualize_logs(loaded_logs, graphs, graphs_ylims,
                           args.nb_plots_per_fig, writer=writer,
                           xlim=args.xlim,
                           remove_outliers=args.remove_outliers)
    else:
        visualize_logs(loaded_logs, graphs, graphs_ylims,
                       args.nb_plots_per_fig, xlim=args.xlim,
                       remove_outliers=args.remove_outliers)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
