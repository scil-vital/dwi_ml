# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_loaders import BatchLoaderOneInput


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
        '--noise_gaussian_size', type=float, metavar='s', default=0.,
        help="If set, add random Gaussian noise to streamline coordinates \n"
             "with given variance. This corresponds to the std of the \n"
             "Gaussian. [0]\n**Make sure noise is smaller than your step size "
             "to avoid \nflipping direction! (We can't verify if --step_size "
             "is not \nspecified here, but if it is, we limit noise to \n"
             "+/- 0.5 * step-size.).\n"
             "** We also limit noise to +/- 2 * noise_gaussian_size.\n"
             "Suggestion: 0.1 * step-size.")
    bl_g.add_argument(
        '--noise_gaussian_variability', type=float, metavar='v', default=0.,
        help="If set, a variation is applied to the noise_size to have very \n"
             "noisy streamlines and less noisy streamlines. This means that \n"
             "the real gaussian_size will be a random number in the range \n"
             "[size - variability, size + variability]. [0]")
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


def prepare_batchloadersoneinput_train_valid(dataset, args_t, args_v,
                                             log_level):
    with Timer("\nPreparing batch loaders...", newline=True, color='pink'):
        logging.info("Instantiating training set's batch loader...")
        training_batch_loader = _prepare_batchloader(
            dataset.training_set, args_t, log_level)

        if dataset.validation_set.nb_subjects > 0:
            logging.info("Instantiating validation set's batch loader...")
            validation_batch_loader = _prepare_batchloader(
                dataset.validation_set, args_v, log_level)

        else:
            validation_batch_loader = None

    return training_batch_loader, validation_batch_loader


def _prepare_batchloader(subset, args, log_level):
    if args.step_size and args.step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.

    if args.split_ratio > 1 or args.split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 "
                         "and 1).")

    if args.reverse_ratio > 1 or args.reverse_ratio < 0:
        raise ValueError("reverse_ratio must be a ratio (i.e. between 0 "
                         "and 1).")

    batch_loader = BatchLoaderOneInput(
        subset, input_group_name=args.input_group_name,
        streamline_group_name=args.streamline_group_name,
        # STREAMLINES PREPROCESSING
        step_size=args.step_size, compress=args.compress,
        # STREAMLINES AUGMENTATION
        noise_gaussian_size=args.noise_gaussian_size,
        noise_gaussian_variability=args.noise_gaussian_variability,
        reverse_ratio=args.reverse_ratio, split_ratio=args.split_ratio,
        # NEIGHBORHOOD
        neighborhood_points=args.neighborhood_points,
        # OTHER
        rng=args.rng, wait_for_gpu=args.wait_for_gpu, log_level=log_level)

    logging.info("Loader user-defined parameters: " +
                 format_dict_to_str(batch_loader.params_for_json_prints))
    return batch_loader
