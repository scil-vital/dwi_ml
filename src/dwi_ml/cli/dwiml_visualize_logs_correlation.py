#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Computes the correlation between two logs, for each experiment.
"""
import argparse
import logging
import os
import pathlib

from matplotlib import colors
import matplotlib.cm as cmx
import matplotlib.pyplot as plt
import numpy as np

from scilpy.io.utils import add_overwrite_arg, add_verbose_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiments", nargs='+',
                   help="Path to the experiment folder(s).")
    p.add_argument("log1",
                   help="Name of the first log.")
    p.add_argument("log2",
                   help="Name of the log to correlate with log1.")
    p.add_argument('--rename_log1',
                   help="If set, renames log on the plot.")
    p.add_argument('--rename_log2')
    p.add_argument('--ignore_first_epochs', type=int, metavar='n',
                   help="If set, ignores the first n epochs of each "
                        "experiment.")
    p.add_argument('--show_individual_logs', action='store_true',
                   help="If set, shows individual logs as well as the "
                        "correlation graph (3 graphs total)")
    p.add_argument('--show_first_order_fit', action='store_true',
                   help="If set, shows first order fit.")
    p.add_argument('--show_second_order_fit', action='store_true',
                   help="If set, show the quadratic fit.")
    p.add_argument('--xlim', nargs=2, type=float)
    p.add_argument('--ylim', nargs=2, type=float)

    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def _load_chosen_logs(parser, args, logs_path):
    """
    Load the two logs for all experiments.
    """
    logging.debug("Loading only chosen logs for that experiment")
    this_exp_dict = {}
    for _log in [args.log1, args.log2]:
        file = pathlib.Path(logs_path, _log + '.npy')
        if os.path.isfile(file):
            this_exp_dict[file.stem] = np.load(file)
        else:
            parser.error("File {} not found in path {}.".format(_log, file))

    return this_exp_dict


def _compute_correlations(loaded_dicts, log1_key, log2_key,
                          name_log1, name_log2, first_epoch, xlim, ylim,
                          show_individual,
                          show_first_order, show_second_order):
    # One color per experiment
    jet = plt.get_cmap('jet')
    exp_names = list(loaded_dicts.keys())
    c_norm = colors.Normalize(vmin=0, vmax=len(exp_names))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    all_x = []
    all_y = []
    labels = []
    if show_individual:
        fig, axs = plt.subplots(3, 1)
    else:
        fig, axs = plt.subplots(1, 1)
        axs = [axs]
    for i, exp in enumerate(loaded_dicts.keys()):
        color_val = scalar_map.to_rgba(i)
        x = loaded_dicts[exp][log1_key][first_epoch:]
        y = loaded_dicts[exp][log2_key][first_epoch:]
        epochs = np.arange(first_epoch, len(x) + first_epoch) + 1

        corr = np.corrcoef(x, y)[0][1]
        print("Correlation for exp {} is {}".format(exp, corr))

        if show_individual:
            axs[0].scatter(epochs, x, color=color_val, s=10)
            axs[1].scatter(epochs, y, color=color_val, s=10)
            axs[2].scatter(x, y, color=color_val, s=10)
            axs[0].set_ylabel(name_log1)
            axs[0].set_xlabel("Epochs")
            axs[1].set_ylabel(name_log2)
            axs[1].set_xlabel("Epochs")
            ax_corr = axs[2]
        else:
            axs[0].scatter(x, y, color=color_val, s=10)
            ax_corr = axs[0]
        labels.append(exp + ':{:.2f}'.format(corr))

        all_x.extend(x)
        all_y.extend(y)

    corr = np.corrcoef(all_x, all_y)[0][1]
    idx = np.argsort(all_x)
    all_x = np.asarray(all_x)[idx]
    all_y = np.asarray(all_y)[idx]
    titre = ("Correlation between {} and {} = {:.3f}."
             .format(name_log1, name_log2, corr))
    print("Correlation over all experiments is {}.".format(corr))

    # Linear fitting
    x_line = np.linspace(min(all_x), max(all_x), 200)
    if show_first_order:
        # Fit
        coef_lin = np.polyfit(all_x, all_y, 1)

        # Residuals
        y_lin = np.polyval(coef_lin, all_x)
        res_lin = all_y - y_lin
        mse_lin = np.mean(res_lin**2)

        # Nice plot
        y_lin_line = np.polyval(coef_lin, x_line)
        ax_corr.plot(x_line, y_lin_line, color='k')
        titre += "\nMSE (linear): {:.1e}".format(mse_lin)

    # Quadratic fitting
    if show_second_order:
        # Fit
        coef_quad = np.polyfit(all_x, all_y, 2)

        # Residuals
        y_quad = np.polyval(coef_quad, all_x)
        res_quad = all_y - y_quad
        mse_quad = np.mean(res_quad ** 2)

        # Nice plot
        y_quad_line = np.polyval(coef_quad, x_line)
        ax_corr.plot(x_line, y_quad_line, color='k')
        titre += "\nMSE (quadratic): {:.1e}".format(mse_quad)


    ax_corr.set_xlabel(name_log1)
    ax_corr.set_ylabel(name_log2)
    ax_corr.set_title(titre)
    if xlim:
        ax_corr.set_xlim(*xlim)
    if ylim:
        ax_corr.set_ylim(*ylim)

    # The xlabels and titles overlap
    # plt.tight_layout()  # Makes the subplots very thin.
    # fig.subplots_adjust()  # Does not work. Insead, below, setting h * 0.8

    # Shrink subplots to leave place for legend
    for ax in axs:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height * 0.8])
    fig.legend(labels, loc="center right")
    fig.suptitle("Correlation throughout all epochs, all experiments")
    plt.show()


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(level=args.verbose)

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
        this_exp_dict = _load_chosen_logs(parser, args, logs_path)
        loaded_logs[exp_name] = this_exp_dict

    name_log1 = args.rename_log1 or args.log1
    name_log2 = args.rename_log2 or args.log2
    first_epoch = args.ignore_first_epochs or 0
    _compute_correlations(loaded_logs, args.log1, args.log2,
                          name_log1, name_log2, first_epoch,
                          args.xlim, args.ylim,
                          args.show_individual_logs,
                          args.show_first_order_fit,
                          args.show_second_order_fit)


if __name__ == '__main__':
    main()
