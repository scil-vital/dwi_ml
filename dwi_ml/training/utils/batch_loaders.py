# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.with_generation.batch_loader import \
    DWIMLBatchLoaderWithConnectivity


def add_args_batch_loader(p: argparse.ArgumentParser):
    # STREAMLINES PREPROCESSING
    bl_g = p.add_argument_group("Batch loader")
    bl_g.add_argument(
        '--noise_gaussian_size_forward', type=float, metavar='s', default=0.,
        help="If set, add random Gaussian noise to streamline coordinates \n"
             "with given variance. Noise is added AFTER interpolation of "
             "underlying data. \nExample of use: when concatenating previous "
             "direction to input.\n"
             "This corresponds to the std of the Gaussian. [0]\n"
             "**Make sure noise is smaller than your step size "
             "to avoid \nflipping direction! (We can't verify if --step_size "
             "is not \nspecified here, but if it is, we limit noise to \n"
             "+/- 0.5 * step-size.).\n"
             "** We also limit noise to +/- 2 * noise_gaussian_size.\n"
             "Suggestion: 0.1 * step-size.")
    bl_g.add_argument(
        '--noise_gaussian_size_loss', type=float, metavar='s', default=0.,
        help='Idem, but loss is added to targets instead (during training '
             'only).')
    bl_g.add_argument(
        '--split_ratio', type=float, metavar='r', default=0.,
        help="Percentage of streamlines to randomly split into 2, in each \n"
             "batch (keeping both segments as two independent streamlines). \n"
             "The reason for cutting is to help the ML algorithm to track "
             "from \nthe middle of WM by having already seen half-streamlines."
             "\nIf you are using interface seeding, this is not necessary. "
             "[0]")
    bl_g.add_argument(
        '--reverse_ratio', type=float, metavar='r', default=0.,
        help="Percentage of streamlines to randomly reverse in each batch. "
             "[0]")


def prepare_batch_loader(dataset, model, args, sub_loggers_level):
    # Preparing the batch loader.
    with Timer("\nPreparing batch loader...", newline=True, color='pink'):
        batch_loader = DWIMLBatchLoaderWithConnectivity(
            dataset=dataset, model=model,
            input_group_name=args.input_group_name,
            streamline_group_name=args.streamline_group_name,
            # STREAMLINES AUGMENTATION
            noise_gaussian_size_forward=args.noise_gaussian_size_forward,
            noise_gaussian_size_loss=args.noise_gaussian_size_loss,
            reverse_ratio=args.reverse_ratio, split_ratio=args.split_ratio,
            # OTHER
            rng=args.rng, log_level=sub_loggers_level)

        logging.info("Loader user-defined parameters: " +
                     format_dict_to_str(batch_loader.params_for_checkpoint))

    return batch_loader
