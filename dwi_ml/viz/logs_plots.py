# -*- coding: utf-8 -*-
import logging
from typing import Dict, List

import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.cm as cmx
import numpy as np

log_styles = ['-', '--', '_.', ':']
nb_styles = len(log_styles)


def _plot_one_line(ax, data, color, style, exp_name, remove_outliers,
                   show_legend):
    """
    Simply ax.plot(data) with given styling options.
    With optional outlier removal.
    """
    epochs = np.arange(0, len(data))
    if remove_outliers:
        std3 = 3 * np.std(data)
        mean = np.mean(data)
        min_val = mean - std3
        max_val = mean + std3
        outlier_idx = np.logical_or(data < min_val, data > max_val)

        logging.info("{}: Found {} outliers out of [{}, {}]."
                     .format(exp_name, outlier_idx.sum(),
                             min_val, max_val))
        ax.scatter(epochs[outlier_idx], data[outlier_idx],
                   marker='*', color=color)
        epochs = epochs[~outlier_idx]
        data = data[~outlier_idx]

    if show_legend:
        label = exp_name
    else:
        label = '_no_label'
    ax.plot(epochs, data, linestyle=style, label=label, color=color)


def _plot_one_graph(ax, logs, exp_names, chosen_log_keys, scalar_map,
                    writer, remove_outliers, title, xlim, ylims):

    # For each log to show in that plot
    for i, log_key in enumerate(chosen_log_keys):
        logging.debug(" - log: {}".format(log_key))
        style = log_styles[i % nb_styles]
        show_legend = i == 0  # Show only legend once per experiment.

        # For each experiment to show:
        for j, exp_name in enumerate(exp_names):
            if log_key in logs[exp_name].keys():
                logging.debug(" -    exp: {}".format(exp_name))
                color_val = scalar_map.to_rgba(j)
                data = logs[exp_name][log_key]
                _plot_one_line(ax, data, color_val, style, exp_name,
                               remove_outliers, show_legend)

                if writer is not None:
                    writer.writerow([exp_name, log_key] + list(data))
            else:
                logging.debug("Not adding exp {} for log {}: missing! \n"
                              "   Possible keys were: {}"
                              .format(exp_name, log_key,
                                      list(logs[exp_name].keys())))

    if len(chosen_log_keys) > 1:
        for i, log_key in enumerate(chosen_log_keys):
            style = log_styles[i % nb_styles]
            ax.plot([], [], label='{}: {}'.format(log_key, style))

    ax.set_title(title)
    ax.legend()
    if xlim is not None:
        ax.set_xlim([0, xlim])
    else:
        ax.set_xlim([0, None])
    if ylims is not None:
        ax.set_ylim(ylims)


def visualize_logs(logs_data: Dict[str, Dict[str, np.ndarray]],
                   graphs_titles: List[str], graphs_logs: List[List[str]],
                   graphs_ylim: List, nb_rows: int, writer=None, xlim=None,
                   remove_outliers=False, save_figs: str = None,
                   fig_size: List = None):
    """
    Parameters
    ----------
    logs_data: dict
        Dict of {exp_name: {log_name: data}}
    graphs_titles: List[str]
        For each graph, the title.
    graphs_logs: List
        For each graph, list of logs to plot.
    graphs_ylim:
        For each graph, the 2-valued list [y_min, y_max], or None.
    nb_rows: int
        The number of plots per figure
    writer:
        CSV writer
    xlim: float
        Maximal xlim (minimal is 0)
    remove_outliers: bool
        If true, replaces the ±3 std values with a *
    save_figs: str
        If set, saves the figure in that dir/prefix location
    fig_size: List[int]
        2-valued figure size (x, y)
    """
    if writer is not None:
        writer.writerow(['Experiment name', "Log name", "Epochs..."])

    jet = plt.get_cmap('jet')

    # One color per experiment
    exp_names = list(logs_data.keys())
    c_norm = colors.Normalize(vmin=0, vmax=len(exp_names))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    # Separating graphs in figures:
    nb_plots_left = len(graphs_logs)
    current_fig = -1
    i = -1
    while nb_plots_left > 0:
        # Prepare figure and axes
        next_nb_plots = min(nb_plots_left, nb_rows)
        fig, axs = plt.subplots(nrows=next_nb_plots)
        if fig_size is not None:
            fig.set_figheight(fig_size[0])
            fig.set_figwidth(fig_size[1])
        current_fig += 1
        if next_nb_plots == 1:
            axs = [axs]

        # For each subplot in this figure
        for ax in range(next_nb_plots):
            i += 1
            logging.debug("Next graph: {}".format(graphs_logs[i]))
            _plot_one_graph(axs[ax], logs_data, exp_names, graphs_logs[i],
                            scalar_map, writer, remove_outliers,
                            graphs_titles[i], xlim, graphs_ylim[i])

        plt.tight_layout()
        if save_figs:
            plt.savefig(save_figs + '_plot{}'.format(current_fig))

        nb_plots_left -= next_nb_plots

