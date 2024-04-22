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
        help="If set, we will add random Gaussian noise to the streamline "
             "coordinates. Noise \nis added AFTER interpolation of "
             "the DWI inputs, so this is only useful if your \nforward method "
             "uses the streamlines; when they also serve as inputs. See also\n"
             "noise_gaussian_size_loss for an alternate option. The value "
             "corresponds to the \nstd of the Gaussian. We limit noise to "
             "+/- 2 * noise_gaussian_size. Suggestion: \n0.1 * step-size.\n"
             "**Make sure that this noise is smaller than your step size, "
             "to avoid flipping \ndirection! (If --step_size is set, we limit "
             "noise to +/- 0.5 * step-size). ")
    bl_g.add_argument(
        '--noise_gaussian_size_loss', type=float, metavar='s', default=0.,
        help='Idem, but loss is added to targets instead (during training '
             'only).')
    bl_g.add_argument(
        '--split_ratio', type=float, metavar='r', default=0.,
        help="Percentage of streamlines to randomly split into 2, in each "
             "batch (keeping both \nsegments as two independent streamlines). "
             "The reason for cutting is to help \ntracking from the middle of "
             "white matter by having already seen half-streamlines."
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
