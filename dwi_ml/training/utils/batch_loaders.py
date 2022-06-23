# -*- coding: utf-8 -*-
import argparse


def add_args_batch_loader(p: argparse.ArgumentParser):
    # STREAMLINES PREPROCESSING
    bl_g = p.add_argument_group(
        "Batch loader")
    sub = bl_g.add_mutually_exclusive_group()
    sub.add_argument(
        '--step_size', type=float, metavar='s',
        help="Resample all streamlines to this step size (in mm). If not set,"
             "\nwe will keep streamlines as they are. Note that you may have\n"
             "already resampled or compressed your data when creating your \n"
             "dataset, but you may use a different choice in the batch \n"
             "sampler if you wish.")
    sub.add_argument(
        '--compress', action='store_true',
        help="If set, compress streamlines. Once again, the choice can be \n"
             "different in the batch sampler than chosen when creating the \n"
             "hdf5.")
    bl_g.add_argument(
        '--noise_gaussian_size_training', type=float, metavar='s', default=0.,
        help="If set, add random Gaussian noise to streamline coordinates \n"
             "with given variance. This corresponds to the std of the \n"
             "Gaussian. [0]\n**Make sure noise is smaller than your step size "
             "to avoid \nflipping direction! (We can't verify if --step_size "
             "is not \nspecified here, but if it is, we limit noise to \n"
             "+/- 0.5 * step-size.).\n"
             "** We also limit noise to +/- 2 * noise_gaussian_size.\n"
             "Suggestion: 0.1 * step-size.")
    bl_g.add_argument(
        '--noise_gaussian_size_validation', type=float, metavar='s',
        default=0.,
        help="Idem; noise added during validation.")
    bl_g.add_argument(
        '--noise_gaussian_variability_training', type=float, metavar='v',
        default=0.,
        help="If set, a variation is applied to the noise_size to have very \n"
             "noisy streamlines and less noisy streamlines. This means that \n"
             "the real gaussian_size will be a random number in the range \n"
             "[size - variability, size + variability]. [0]")
    bl_g.add_argument(
        '--noise_gaussian_variability_validation', type=float, metavar='v',
        default=0.,
        help="Idem; will be used during validation.")
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
