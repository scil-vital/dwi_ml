#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained projects model.

See scilpy's scil_compute_local_tracking and scil_compute_local_tracking_2
for references.
"""
import argparse
import logging
import math

import dipy.core.geometry as gm
from dipy.io.stateful_tractogram import (StatefulTractogram, Space,
                                         set_sft_logger_level, Origin)
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
import h5py
import nibabel as nib
import numpy as np
import torch

from scilpy.io.utils import (add_sphere_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.data.dataset.utils import add_dataset_args
from dwi_ml.experiment_utils.prints import add_logging_arg, format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.tracking.utils import (add_mandatory_options_tracking,
                                   add_tracking_options,
                                   prepare_dataset_for_tracking,
                                   prepare_seed_generator,
                                   prepare_tracking_mask,
                                   prepare_step_size_vox)

##################
# PLEASE COPY AND ADAPT:
##################
# Use your own model.
from dwi_ml.tests.utils.data_and_models_for_tests import (
    TrackingModelForTestWithPD, TestPropagator)

# Choose appropriate classes or implement your own child classes.
# Example:
from dwi_ml.tracking.tracker import DWIMLTracker


def build_argparser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    # If you need the sphere for your model:
    add_sphere_arg(track_g, symmetric_only=False)

    add_dataset_args(p, with_volumes=True)

    # As in scilpy:
    add_seeding_options(p)
    add_out_options(p)

    add_logging_arg(p)

    return p


def prepare_tracker(parser, args, device, min_nbr_pts, max_nbr_pts,
                    max_invalid_dirs):
    hdf_handle = h5py.File(args.hdf5_file, 'r')

    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'

    with Timer("\nLoading data and preparing tracker...",
               newline=True, color='green'):
        logging.info("Loading seeding mask + preparing seed generator.")
        seed_generator, nbr_seeds, seeding_mask_header = \
            prepare_seed_generator(parser, args, hdf_handle)

        logging.info("Loading tracking mask.")
        tracking_mask, ref = prepare_tracking_mask(args, hdf_handle)

        # Comparing tracking and seeding masks
        is_header_compatible(ref, seeding_mask_header)
        res = seeding_mask_header['pixdim'][0:3]

        logging.info("Loading subject's data.")
        subset, subj_idx = prepare_dataset_for_tracking(args)

        logging.info("Loading model.")
        model = TrackingModelForTestWithPD.load_params_and_state(
            args.experiment_path + '/best_model', log_level=sub_logger_level)
        logging.info("* Loaded params: " +
                     format_dict_to_str(model.params_for_json_prints) +
                     "\n")

        logging.debug("Instantiating propagator.")
        step_size_vox, normalize_directions = prepare_step_size_vox(
            args.step_size, res)
        theta = gm.math.radians(args.theta)
        propagator = TestPropagator(
            input_volume_group=args.input_group,
            dataset=subset, subj_idx=subj_idx, model=model,
            step_size=step_size_vox, algo=args.algo, theta=theta,
            device=device, normalize_directions=normalize_directions,
            verify_opposite_direction=False)

        logging.debug("Instantiating tracker.")
        tracker = DWIMLTracker(
            propagator, tracking_mask, seed_generator, nbr_seeds, min_nbr_pts,
            max_nbr_pts, max_invalid_dirs, args.compress, args.nbr_processes,
            args.save_seeds, args.rng_seed, args.track_forward_only,
            use_gpu=args.use_gpu,
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
    # Using step size in mm here.
    max_nbr_pts = int(args.max_length / args.step_size)
    min_nbr_pts = int(args.min_length / args.step_size) + 1
    max_invalid_dirs = int(math.ceil(args.max_invalid_len / args.step_size))

    device = torch.device('cpu')
    if args.use_gpu:
        if args.nbr_processes > 1:
            logging.warning("Number of processes was set to {} but you "
                            "are using GPU. Parameter ignored."
                            .format(args.nbr_processes))
        if torch.cuda.is_available():
            device = torch.device('cuda')

    tracker, ref = prepare_tracker(parser, args, device, min_nbr_pts,
                                   max_nbr_pts, max_invalid_dirs)

    # ----- Track

    with Timer("\nTracking...", newline=True, color='blue'):
        streamlines, seeds = tracker.track()

        logging.debug("Tracked {} streamlines (out of {} seeds). Now saving..."
                      .format(len(streamlines), tracker.nbr_seeds))

    if len(streamlines) == 0:
        logging.warning("No streamlines created! Not saving tractogram!")
        return

    # save seeds if args.save_seeds is given
    # We seeded (and tracked) in vox, corner, but in dipy, they use
    # vox, center. In other scripts using the seeds (ex,
    # scil_compute_density_map), this is what they will expect.
    if args.save_seeds:
        print("Saving seeds in data_per_streamline.")
        seeds = [np.asanyarray(seed) - 0.5 for seed in seeds]  # to_center
        data_per_streamline = {'seeds': seeds}
    else:
        data_per_streamline = {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    sft = StatefulTractogram(streamlines, ref, Space.VOX,
                             origin=Origin.TRACKVIS,
                             data_per_streamline=data_per_streamline)

    print("Saving tractogram {} with {} streamlines"
          .format(args.out_tractogram, len(sft.streamlines)))
    save_tractogram(sft, args.out_tractogram,
                    bbox_valid_check=False)


if __name__ == "__main__":
    main()
