#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Plot your logs.

All given experiments are shown together, with one color per experiment. By
default, all logs are shown in a separate graph, up to 3 graphs per figure.

The number of graphs per figure can be modified with --nb_plots_per_fig.

You may also specify the logs you want to plot. Use the option --graph, with
the name(s) of the logs to add to one graph. This option can be used many
times. Ex:
>> --graph train_loss_monitor_per_epoch
>> --graph log1 --graph log2 log3

Optionally, you can add the 2-valued ylims for each graph. These value will
supersede the given --ylim, if any.
>> --graph log1 0 100

Finally, you can also supply operations to apply to your logs, amongst:
['diff', 'sum']. Ex:
>> --graph diff(log1, log2) 0 100
** Note that we only accept one operation per graph. The following is not
supported:
>> --graph diff(log1, log2) log 3 0 100

------------------------------
"""
import argparse
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
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiments", type=str, nargs='+',
                   help="Path to the experiment folder (s). If more than "
                        "one, \nthey are shown superposed.")
    p.add_argument("--save_to_csv", metavar='my_file.csv',
                   help="If set, convert the resulting logs as a csv file.")
    p.add_argument('--save_figures', metavar='folder/prefix', default='./',
                   help="If set, saves the resulting figures in chosen "
                        "folder. Default: ./out_\n"
                        "Figure names are: fig1, fig2, etc.")
    p.add_argument('--allow_missing', action='store_true',
                   help="If true, ignore experiments with missing logs.")

    g = p.add_argument_group("Figure options")
    g.add_argument("--graph", action='append', nargs='+', dest='graphs',
                   help="See description above for usage.")
    g.add_argument("--nb_plots_per_fig", type=int, default=3, metavar='n',
                   help="Number of (rows) of plot per figure. Default: 3.")
    g.add_argument('--xlim', type=int, metavar='epoch_max',
                   help="All graphs' xlim.")
    g.add_argument('--ylims', type=float, nargs=2, metavar='ymin ymax',
                   help="All graph's ylim. (Makes little sense with more "
                        "than one graph.)")

    g = p.add_argument_group("Processing options")
    g.add_argument('--remove_outliers', action='store_true',
                   help="If set, remove outliers (3 std) from plot line and "
                        "adds a star \nsign instead. Outliers are not removed "
                        "in the CSV file, if any.")

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def _parse_graphs_arg(parser, args):
    """Parse args.graphs"""
    if args.graphs is None:
        return None, None, None
    else:
        graphs = []
        graphs_ylims = []
        graph_operations = []
        for graph in args.graphs:
            # Verify if gave the ylims.
            if len(graph) > 1:
                try:
                    min_y = float(graph[-2])
                    max_y = float(graph[-1])
                    _logs = graph[:-2]
                    _ylims = [min_y, max_y]
                except ValueError:
                    try:
                        _ = float(graph[-1])
                        parser.error("For option --graph {}: you seem to have "
                                     "added a single float value at the end. "
                                     "Ylim requires two values.".format(graph))
                    except ValueError:
                        # There does not seem to be a ylim.
                        _logs = graph
                        _ylims = None
            else:
                _logs = graph
                _ylims = None

            # Verify in gave an operation
            _logs, operation = __parse_log_operations(parser, _logs)

            # Remove .npy to log names if added by user.
            for i, log in enumerate(_logs):
                f, ext = os.path.splitext(log)
                if ext not in ['.npy', '']:
                    parser.error("The values to the --graph option "
                                 "should be log names, but got {} (try "
                                 "removing the extension).".format(log))
                _logs[i] = f

            # All ok
            graphs.append(_logs)
            graphs_ylims.append(_ylims)
            graph_operations.append(operation)

        return graphs, graphs_ylims, graph_operations


def __parse_log_operations(parser, graph):
    if len(graph) == 1:
        _graph = graph[0]
        if _graph[0:5] == 'diff(':
            assert _graph[-1] == ')'
            _graph = _graph[5:-1]
            logs = _graph.split(',')
            if len(logs) != 2:
                parser.error("Can't understand option --graph graph. The diff "
                             "operation requires two logs, separated by a "
                             "comma.")
            logs = [l.replace(' ', '') for l in logs]
            return logs, 'diff'
        elif _graph[0:4] == 'sum(':
            assert _graph[-1] == ')'
            _graph = _graph[4:-1]
            logs = _graph.split(',')
            if len(logs) != 2:
                parser.error("Can't understand option --graph graph. The sum "
                             "operation requires two logs, separated by a "
                             "comma.")
            logs = [l.replace(' ', '') for l in logs]
            return logs, 'sum'

    op = None
    return graph, op


def _load_all_logs(parser, args, logs_path, previous_graphs):
    """
    Load all logs in an experiment's dir (no option --graph was supplied)
    """
    logging.debug("Loading all logs for that experiment.")
    files_to_load = list(logs_path.glob('*.npy'))
    _graphs = [f.stem for f in files_to_load]

    if previous_graphs is None:
        # Using the first experiment to set the list of expected logs.
        previous_graphs = _graphs
    else:
        if args.allow_missing:
            # Combine all log names from all experiments.
            previous_graphs = list(set(previous_graphs.extend(_graphs)))
        else:
            # Compare with first experiment's log
            if set(previous_graphs) != set(_graphs):
                parser.error(
                    "The experiments do not all have the same logs. "
                    "To allow running, use --allow_missing.")

    # Load
    this_exp_dict = {}
    for file in files_to_load:
        this_exp_dict[file.stem] = np.load(file)

    return previous_graphs, this_exp_dict


def _load_chosen_logs(parser, args, logs_path, parsed_graphs,
                      graph_operations):
    """
    Load only logs specified through --graph
    """
    logging.debug("Loading only chosen logs for that experiment")
    this_exp_dict = {}
    for i, required_logs in enumerate(parsed_graphs):
        if graph_operations[i] is None:
            # Simpy add each log.
            for required_log in required_logs:
                file = pathlib.Path(logs_path, required_log + '.npy')
                if os.path.isfile(file):
                    if file not in this_exp_dict.keys():
                        this_exp_dict[file.stem] = np.load(file)
                else:
                    if not args.allow_missing:
                        parser.error(
                            "File {} not found in path {}."
                            .format(required_log, file))
                    else:
                        logging.debug("Exp {} will not show on --graph {}, "
                                      "because we could not find its logs."
                                      .format(logs_path, required_logs))
        else:
            # Apply operation. All our operations require two values.
            op = graph_operations[i]
            file1 = pathlib.Path(logs_path, required_logs[0] + '.npy')
            file2 = pathlib.Path(logs_path, required_logs[1] + '.npy')
            if not os.path.isfile(file1) or not os.path.isfile(file2):
                if not args.allow_missing:
                    parser.error(
                        "File {} and/or {} not found in path {}."
                        .format(required_logs[0], required_logs[1], logs_path))
                else:
                    logging.debug("Exp {} will not show on --graph {}({}), "
                                  "because we could not find its logs."
                                  .format(logs_path, op, required_logs))
            else:
                data1 = np.load(file1)
                data2 = np.load(file2)
                if len(data1) != len(data2):
                    parser.error("Unexpected error. logs {} and {} do not "
                                 "have the same length!?! Cannot use "
                                 "operation {}"
                                 .format(required_logs[0], required_logs[1],
                                         op))
                if op == 'diff':
                    data = data1 - data2
                else:
                    data = data1 + data2
                new_name = _format_name_after_operation(
                    required_logs[0], required_logs[1], op)
                this_exp_dict[new_name] = data
    return this_exp_dict


def _format_name_after_operation(name1, name2, operation):
    if operation == 'diff':
        name = name1 + '-' + name2
    else:  # operation == 'sum':
        name = name1 + '+' + name2
    return name


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
    parsed_graphs, graphs_ylims, graphs_operation = \
        _parse_graphs_arg(parser, args)
    if args.save_figures is not None:
        if os.path.isdir(args.save_figures):
            args.save_figures = os.path.join(args.save_figures, 'out_')
        else:
            # Maybe a prefix was added?
            fig_path, prefix = os.path.split(args.save_figures)
            if not os.path.isdir(fig_path):
                parser.error("Output dir for figures does not exist.")

    # Loop on all experiments
    loaded_logs = {}  # dict of dicts
    for i, exp_path in enumerate(args.experiments):
        # Verifications for this experiment
        if not pathlib.Path(exp_path).exists():
            raise ValueError("Experiment folder does not exist: {}"
                             .format(exp_path))
        exp_name = os.path.basename(os.path.normpath(exp_path))
        logs_path = pathlib.Path(exp_path, 'logs')
        if not logs_path.exists():
            raise ValueError("Logs folder does not exist for experiment {}!"
                             .format(exp_path))

        # Loading
        logging.info("Loading logs from experiment {}.".format(exp_name))
        if args.graphs is None:
            parsed_graphs, this_exp_dict = _load_all_logs(
                parser, args, logs_path, parsed_graphs)

        else:
            this_exp_dict = _load_chosen_logs(parser, args, logs_path,
                                              parsed_graphs, graphs_operation)
        loaded_logs[exp_name] = this_exp_dict

    # Formatting the final graphs choice.
    if args.graphs is None:
        parsed_graphs = [[log] for log in parsed_graphs]
        graphs_ylims = [None for _ in parsed_graphs]
        # graphs_operation = [None for _ in parsed_graphs]
    else:
        parsed_graphs = [g if op is None else
                         [_format_name_after_operation(*g, op)]
                         for g, op in zip(parsed_graphs, graphs_operation)]

    # Final formatting of ylims, for graphs that did have another choice.
    if args.ylims:
        graphs_ylims = [args.ylims if ylim is None else ylim
                        for ylim in graphs_ylims]

    if args.save_to_csv:
        logging.info("Will save results in file {}".format(args.save_to_csv))
        with open(args.save_to_csv, 'w', newline='') as file:
            writer = csv.writer(file)
            visualize_logs(loaded_logs, parsed_graphs, graphs_ylims,
                           args.nb_plots_per_fig, writer=writer,
                           xlim=args.xlim,
                           remove_outliers=args.remove_outliers,
                           save_figs=args.save_figures)
    else:
        visualize_logs(loaded_logs, parsed_graphs, graphs_ylims,
                       args.nb_plots_per_fig,
                       xlim=args.xlim, remove_outliers=args.remove_outliers,
                       save_figs=args.save_figures)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
