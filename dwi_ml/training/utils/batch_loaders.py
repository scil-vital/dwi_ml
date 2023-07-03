# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.with_generation.batch_loader import \
    DWIMLBatchLoaderWithConnectivity


def get_args_batch_loader():
    args = {
        '--noise_gaussian_size_forward': {
            'type': float, 'metavar': 's', 'default': 0.,
            'help': "If set, adds random Gaussian noise to streamline "
                    "coordinates. This corresponds \nto the std of the "
                    "Gaussian. Default: 0.0 (no noise). Suggestion: 0.1 * "
                    "step-size.\n"
                    "**Make sure noise is smaller than your step size to "
                    "avoid flipping direction! \n"
                    "**We also limit noise to +/- 2 * std."},
        '--split_ratio': {
            'type': float, 'metavar': 'r', 'default': 0.,
            'help': "Ratio (percentage) of streamlines to randomly split "
                    "into two segments in each \nbatch (keeping both segments "
                    "as two independent streamlines). \n"
                    "**Hint: The reason for cutting is to help the ML "
                    "algorithm to track from the middle \nof WM by having "
                    "already seen half-streamlines. If you are using "
                    "interface \nseeding, this is probably not necessary. "
                    "Default: 0.0."},
        '--reverse_ratio': {
            'type': float, 'metavar': 'r', 'default': 0.,
            'help': "Ratio (percentage) of streamlines to randomly reverse in "
                    "each batch. Default: 0.0."}
    }
    return args


def prepare_batch_loader(dataset, model, args, sub_loggers_level):
    # Preparing the batch loader.
    with Timer("\nPreparing batch loader...", newline=True, color='pink'):
        batch_loader = DWIMLBatchLoaderWithConnectivity(
            dataset=dataset, model=model,
            input_group_name=args.input_group_name,
            streamline_group_name=args.streamline_group_name,
            # STREAMLINES AUGMENTATION
            noise_gaussian_size_forward=args.noise_gaussian_size_forward,
            reverse_ratio=args.reverse_ratio, split_ratio=args.split_ratio,
            # OTHER
            rng=args.rng, log_level=sub_loggers_level)

        logging.info("Loader user-defined parameters: " +
                     format_dict_to_str(batch_loader.params_for_checkpoint))

    return batch_loader
