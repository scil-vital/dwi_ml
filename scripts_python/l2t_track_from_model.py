#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Learn2track model.
"""
import argparse
import logging
import os

import dipy.core.geometry as gm
from dipy.io.utils import is_header_compatible
import h5py
import nibabel as nib
import torch

from scilpy.io.utils import (add_sphere_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_verbose_arg
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.testing.utils import prepare_dataset_one_subj, \
    find_hdf5_associated_to_experiment
from dwi_ml.tracking.projects.learn2track_tracker import RecurrentTracker
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.tracking.io_utils import (add_tracking_options,
                                      prepare_seed_generator,
                                      prepare_tracking_mask, track_and_save)


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    track_g = add_tracking_options(p)
    # Sphere used if the direction_getter key is the sphere-classification.
    add_sphere_arg(track_g, symmetric_only=False)

    # As in scilpy:
    add_seeding_options(p)
    add_out_options(p)  # Formatting a bit ugly compared to us, but ok.

    add_verbose_arg(p)

    return p


def prepare_tracker(parser, args):
    hdf5_file = args.hdf5_file or find_hdf5_associated_to_experiment(
        args.experiment_path)
    hdf_handle = h5py.File(hdf5_file, 'r')

    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    with Timer("\nLoading data and preparing tracker...",
               newline=True, color='green'):
        logging.info("Loading seeding mask + preparing seed generator.")
        # Vox space, corner origin
        seed_generator, nbr_seeds, seeding_mask_header, ref = \
            prepare_seed_generator(parser, args, hdf_handle)
        dim = ref.shape

        if args.tracking_mask_group is not None:
            logging.info("Loading tracking mask.")
            tracking_mask, ref2 = prepare_tracking_mask(
                hdf_handle, args.tracking_mask_group, args.subj_id,
                args.mask_interp)

            # Comparing tracking and seeding masks
            is_header_compatible(ref2, seeding_mask_header)
        else:
            tracking_mask = TrackingMask(dim)

        logging.info("Loading subject's data.")
        subset = prepare_dataset_one_subj(
            hdf5_file, args.subj_id, lazy=False,
            cache_size=args.cache_size, subset_name=args.subset,
            volume_groups=[args.input_group], streamline_groups=[])

        logging.info("Loading model.")
        if args.use_latest_epoch:
            model_dir = os.path.join(args.experiment_path, 'best_model')
        else:
            model_dir = os.path.join(args.experiment_path, 'checkpoint/model')
        model = Learn2TrackModel.load_model_from_params_and_state(
            model_dir, log_level=sub_loggers_level)
        logging.info("* Formatted model: " +
                     format_dict_to_str(model.params_for_checkpoint))

        theta = gm.math.radians(args.theta)
        logging.debug("Instantiating tracker.")
        append_last_point = not args.discard_last_point
        tracker = RecurrentTracker(
            input_volume_group=args.input_group,
            dataset=subset, subj_idx=0, model=model, mask=tracking_mask,
            seed_generator=seed_generator, nbr_seeds=nbr_seeds,
            min_len_mm=args.min_length, max_len_mm=args.max_length,
            compression_th=args.compress_th, nbr_processes=args.nbr_processes,
            save_seeds=args.save_seeds, rng_seed=args.rng_seed,
            track_forward_only=args.track_forward_only,
            step_size_mm=args.step_size, algo=args.algo, theta=theta,
            use_gpu=args.use_gpu, eos_stopping_thresh=args.eos_stop,
            simultaneous_tracking=args.simultaneous_tracking,
            append_last_point=append_last_point,
            log_level=args.verbose)

    return tracker, ref


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Setting root logger to high level to max info, not debug, prints way too
    # much stuff. (but we can set our tracker's logger to debug)
    logging.getLogger().setLevel(level=args.verbose)

    # ----- Checks
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or '
                     'tck): {0}'.format(args.out_tractogram))

    assert_inputs_exist(parser, [], args.hdf5_file)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress_th)
    verify_seed_options(parser, args)

    tracker, ref = prepare_tracker(parser, args)

    # ----- Track
    track_and_save(tracker, args, ref)


if __name__ == "__main__":
    main()
