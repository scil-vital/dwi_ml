#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from dwi_ml.training.projects.learn2track_trainer import Learn2TrackTrainer


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiments_path',
                   help='Path from where to load your experiment, and where to'
                        'save new results.\nComplete path will be '
                        'experiments_path/experiment_name.')
    p.add_argument('experiment_name',
                   help='If given, name for the experiment.')
    p.add_argument('--feature_names', nargs='*',
                   help="Ex: --feature_name T1 1 FA 2 fODF 3 PD 48")
    return p


def _parse_f_names(args, nb_features):
    f_names = []
    f_starts = []
    if args.feature_names is not None:
        # 1. Parse the information.
        if len(args.feature_names) % 2 != 0:
            raise ValueError("--feature_names should have a even number of "
                             "values.")
        for i in range(0, len(args.feature_names), 2):
            f_names.append(args.feature_names[i])
            f_starts.append(int(args.feature_names[i + 1]))

        # 2. Sort the values
        f_starts, order = np.sort(f_starts), np.argsort(f_starts)
        f_names = np.asarray(f_names)[order]

        if f_starts[-1] > nb_features:
            raise ValueError("One --feature_names started at a value larger "
                             "than {}, the number of features."
                             .format(nb_features))
        f_starts = np.append(f_starts, nb_features)

    return f_names, f_starts


def _prepare_figure(weights, weights_pd, best_epoch, f_names, f_starts,
                    main_title):
    if weights_pd is not None:
        weights = np.hstack((weights, weights_pd))
    nb_epochs = weights.shape[0]
    total_width = weights.shape[1]

    # Preparing a red square around the best model. Can't use the same twice.
    best_epoch_rects = []
    for i in range(2):
        best_epoch_rects.append(patches.Rectangle(
            (-0.5, best_epoch - 0.5), total_width + 1, 1,
            linewidth=1, edgecolor='r', facecolor='none'))

    # Preparing a pink square around each feature group.
    if len(f_names) > 0:
        f_rects = [[], []]

        if weights_pd is not None and weights_pd.size > 0:
            f_names = np.append(f_names, 'Prev. Dirs')
            f_starts = np.append(f_starts, total_width - weights_pd.shape[1])

        # Prepare rectangles.
        for f in range(len(f_names)):
            f_width = f_starts[f + 1] - f_starts[f]
            for i in range(2):
                f_rects[i].append(patches.Rectangle(
                    (f_starts[f] - 0.5, -0.5), f_width, nb_epochs,
                    linewidth=1, edgecolor='pink', facecolor='none'))
        title_suffix = "\nFeatures, in order: {}".format(f_names)
    else:
        title_suffix = ''

    # Now prepare the figure, subplot 1: raw
    fig, ax = plt.subplots(1, 2)
    im1 = ax[0].imshow(weights, aspect='auto', interpolation='none')
    ax[0].set_title("Weights: columns = features, rows = epoch.\n"
                    "In red: best model." + title_suffix)

    # Add colorbar
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im1, cax=cax, orientation='vertical')

    # Suplot 2: normalized per epoch.
    nb_weights = weights.shape[1]
    norm_weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
    norm_weights *= nb_weights
    im2 = ax[1].imshow(norm_weights, aspect='auto', interpolation='none',
                       vmin=0)
    ax[1].set_title("Percentage of the weights (per epoch).")
    ax[1].set_title("Part associated to each weight (per epoch).\n"
                    "1 everywhere = All features have the same importance.")

    # Add colorbar
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')

    # Best epoch rectangle
    for s in range(2):
        ax[s].add_patch(best_epoch_rects[s])
    ax[1].text(total_width + 5, best_epoch, "Best epoch",
               bbox=dict(boxstyle="larrow, pad=0.1"))

    # Features rectangles
    if len(f_names) > 0:
        for s in range(2):
            for f in range(len(f_names)):
                ax[s].add_patch(f_rects[s][f])

    fig.suptitle(main_title)


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Verify if a checkpoint has been saved.
    checkpoint_path = os.path.join(
            args.experiments_path, args.experiment_name, "checkpoint")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Experiment's checkpoint not found ({})."
                                .format(checkpoint_path))

    # Loading checkpoint
    checkpoint_state = Learn2TrackTrainer.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)

    all_weights = checkpoint_state['current_states']['weights_vizualizor']
    best_epoch = checkpoint_state['current_states']['best_epoch_monitoring_state']['best_epoch']

    del checkpoint_state

    # Verifying parameters
    weights_f1 = np.vstack(all_weights['features_current_point'])
    nb_features = weights_f1.shape[1]
    print("Number of features: {}".format(nb_features))

    # Verify neighborhood and nb features
    if len(all_weights['features_mean_neighborhood']) > 0:
        weights_fn = np.vstack(all_weights['features_mean_neighborhood'])
        weights_n = np.vstack(all_weights['neighborhood_mean_features'])
        print("Number of neighbors: {}".format(weights_n.shape[1]))
        assert weights_fn.shape[1] == nb_features
    else:
        weights_fn = None
        weights_n = None
    if len(all_weights['previous_dirs']) > 0:
        weights_pd = np.vstack(all_weights['previous_dirs'])
        print("Infos added on graph for the previous directions: {}"
              .format(weights_pd.shape[1]))
    else:
        weights_pd = None
    weights_hidden_state = np.asarray(all_weights['hidden_state'])

    # Parse f_names
    f_names, f_starts = _parse_f_names(args, nb_features)

    # Fig 1 : weights at current position
    _prepare_figure(weights_f1, weights_pd, best_epoch, f_names, f_starts,
                    "Feature importance at current position.")

    # Fig 2: weights averaged in neighbohood
    if weights_fn is not None:
        _prepare_figure(weights_fn, weights_pd, best_epoch, f_names, f_starts,
                        "Feature importance, averaged in neighborhood.")

    # Fig 3: Importance of neighborhood points.
    if weights_fn is not None:
        f_names = ['Neighb. points']
        f_starts = [0, weights_fn.shape[1]]
        _prepare_figure(weights_n, None, best_epoch, f_names, f_starts,
                        "Neighborhood importance, averaged over features.")

    # Fig 4. Hidden weights
    plt.figure()
    plt.plot(weights_hidden_state)
    plt.vlines(best_epoch,
               weights_hidden_state.min(), weights_hidden_state.max())
    plt.title("Average of hidden weights")

    # Show all
    plt.show()


if __name__ == '__main__':
    main()
