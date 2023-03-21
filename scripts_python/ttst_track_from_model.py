#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Transformer model.
 (original version)
"""
import argparse
import logging
import math

import dipy.core.geometry as gm
from dipy.io.utils import is_header_compatible
import h5py
import nibabel as nib

from scilpy.io.utils import (add_sphere_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.data.dataset.utils import add_dataset_args
from dwi_ml.experiment_utils.prints import format_dict_to_str, add_logging_arg
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.projects.transforming_tractography import TransformerSrcAndTgtModel
from dwi_ml.tracking.projects.transformer_tracker import \
    TransformerTracker
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.tracking.utils import (add_mandatory_options_tracking,
                                   add_tracking_options,
                                   prepare_seed_generator,
                                   prepare_tracking_mask,
                                   prepare_dataset_one_subj,
                                   prepare_step_size_vox, track_and_save)


def build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    # Sphere used if the direction_getter key is the sphere-classification.
    add_sphere_arg(track_g, symmetric_only=False)

    add_dataset_args(p)

    # As in scilpy:
    add_seeding_options(p)
    add_out_options(p)

    add_logging_arg(p)

    return p


def prepare_tracker(parser, args, min_nbr_pts, max_nbr_pts, max_invalid_dirs):
    hdf_handle = h5py.File(args.hdf5_file, 'r')

    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'

    with Timer("\n\nLoading data and preparing tracker...",
               newline=True, color='green'):
        logging.info("Loading seeding mask + preparing seed generator.")
        seed_generator, nbr_seeds, seeding_mask_header, ref = \
            prepare_seed_generator(parser, args, hdf_handle)
        res = seeding_mask_header['pixdim'][0:3]
        dim = ref.shape

        if args.tracking_mask_group is not None:
            logging.info("Loading tracking mask.")
            tracking_mask, ref2 = prepare_tracking_mask(args, hdf_handle)

            # Comparing tracking and seeding masks
            is_header_compatible(ref2, seeding_mask_header)
        else:
            tracking_mask = TrackingMask(dim)

        logging.info("Loading subject's data.")
        subset, subj_idx = prepare_dataset_one_subj(args)

        logging.info("Loading model.")
        model = TransformerSrcAndTgtModel.load_params_and_state(
            args.experiment_path + '/best_model', log_level=sub_logger_level)
        logging.info("* Formatted model: " +
                     format_dict_to_str(model.params_for_checkpoint))

        theta = gm.math.radians(args.theta)
        step_size_vox, normalize_directions = prepare_step_size_vox(
            args.step_size, res)

        logging.debug("Instantiating tracker.")
        tracker = TransformerTracker(
            input_volume_group=args.input_group,
            dataset=subset, subj_idx=subj_idx, model=model, mask=tracking_mask,
            seed_generator=seed_generator, nbr_seeds=nbr_seeds,
            min_nbr_pts=min_nbr_pts, max_nbr_pts=max_nbr_pts,
            max_invalid_dirs=max_invalid_dirs, compression_th=args.compress,
            nbr_processes=args.nbr_processes, save_seeds=args.save_seeds,
            rng_seed=args.rng_seed, track_forward_only=args.track_forward_only,
            step_size=step_size_vox, algo=args.algo, theta=theta,
            normalize_directions=normalize_directions, use_gpu=args.use_gpu,
            simultanenous_tracking=args.simultaneous_tracking,
            log_level=args.logging)

    return tracker, ref


def main():
    parser = build_argparser()
    args = parser.parse_args()

    # Setting root logger to high level to max info, not debug, prints way too
    # much stuff. (but we can set our tracker's logger to debug)
    root_level = args.logging
    if root_level == logging.DEBUG:
        root_level = logging.INFO
    logging.getLogger().setLevel(level=root_level)

    # ----- Checks
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or '
                     'tck): {0}'.format(args.out_tractogram))

    assert_inputs_exist(parser, args.hdf5_file)
    assert_outputs_exist(parser, args, args.out_tractogram)

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    # ----- Prepare values
    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = int(args.min_length / args.step_size)
    max_invalid_dirs = int(math.ceil(args.max_invalid_len / args.step_size)) - 1

    tracker, ref = prepare_tracker(parser, args,
                                   min_nbr_pts, max_nbr_pts, max_invalid_dirs)

    # ----- Track
    track_and_save(tracker, args, ref)


if __name__ == "__main__":
    main()
