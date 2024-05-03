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

from scilpy.io.utils import add_overwrite_arg

from dwi_ml.io_utils import add_verbose_arg


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
                          name_log1, name_log2, first_epoch):
    # One color per experiment
    jet = plt.get_cmap('jet')
    exp_names = list(loaded_dicts.keys())
    c_norm = colors.Normalize(vmin=0, vmax=len(exp_names))
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=jet)

    all_x = []
    all_y = []
    labels = []
    fig, axs = plt.subplots(3, 1)
    for i, exp in enumerate(loaded_dicts.keys()):
        color_val = scalar_map.to_rgba(i)
        x = loaded_dicts[exp][log1_key][first_epoch:]
        y = loaded_dicts[exp][log2_key][first_epoch:]
        epochs = np.arange(first_epoch, len(x) + first_epoch) + 1

        corr = np.corrcoef(x, y)[0][1]
        print("Correlation for exp {} is {}".format(exp, corr))

        axs[0].scatter(epochs, x, color=color_val, s=10)
        axs[1].scatter(epochs, y, color=color_val, s=10)
        axs[2].scatter(x, y, color=color_val, s=10)
        labels.append(exp + ':{:.2f}'.format(corr))

        all_x.extend(x)
        all_y.extend(y)

    corr = np.corrcoef(all_x, all_y)[0][1]
    b, m = np.polynomial.polynomial.polyfit(all_x, all_y, 1)
    xx = np.linspace(np.min(all_x), np.max(all_x), 100)
    axs[2].plot(xx, b + m * xx, color='k',
                label="y={:.4f}x + {:.4f}".format(m, b))
    axs[2].legend()

    print("Correlation over all experiments is {}".format(corr))
    axs[0].set_ylabel(name_log1)
    axs[0].set_xlabel("Epochs")
    axs[1].set_ylabel(name_log2)
    axs[1].set_xlabel("Epochs")
    axs[2].set_xlabel(name_log1)
    axs[2].set_ylabel(name_log2)
    axs[2].set_title("Correlation between {} and {} = {:.4f}"
                     .format(name_log1, name_log2, corr))

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
                          name_log1, name_log2, first_epoch)


if __name__ == '__main__':
    main()
