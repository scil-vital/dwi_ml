#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained projects model.
"""
import argparse
import logging
import math

import dipy.core.geometry as gm
from dipy.io.stateful_tractogram import (StatefulTractogram, Space,
                                         set_sft_logger_level, Origin)
from dipy.io.streamline import save_tractogram
import h5py
import nibabel as nib
import numpy as np
import torch

from scilpy.image.datasets import DataVolume
from scilpy.io.utils import (add_sphere_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.seed import SeedGenerator
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.data.dataset.utils import add_dataset_args
from dwi_ml.experiment_utils.prints import format_dict_to_str, add_logging_arg
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.tracking.projects.learn2track_propagator import RecurrentPropagator
from dwi_ml.tracking.tracker import DWIMLTracker
from dwi_ml.tracking.utils import (add_mandatory_options_tracking,
                                   add_tracking_options,
                                   prepare_dataset_for_tracking)


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


def prepare_tracker(parser, args, hdf5_file, device,
                    min_nbr_pts, max_nbr_pts, max_invalid_dirs):
    hdf_handle = h5py.File(hdf5_file, 'r')

    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'

    with Timer("\nLoading data and preparing tracker...",
               newline=True, color='green'):
        logging.info("Loading seeding mask + preparing seed generator.")
        seed_generator, nbr_seeds, seed_res, seed_data = \
            _prepare_seed_generator(parser, args, hdf_handle)

        logging.info("Loading tracking mask.")
        mask, ref, mask_res, mask_data = \
            _prepare_tracking_mask(args, hdf_handle)

        # Comparing tracking and seeding masks
        if not mask_res[0] == mask_res[1] == mask_res[2]:
            raise NotImplementedError("This is not ready for anisotropic "
                                      "resolution.")

        if not np.array_equal(mask_res, seed_res):
            parser.error("Not the same resolution for the tracking and "
                         "seeding masks! ({} vs {})".format(mask_res,
                                                            seed_res))
        if not np.array_equal(mask_data.shape, seed_data.shape):
            parser.error("Not the same data size for the tracking and "
                         "seeding masks! ({} vs {})"
                         .format(mask_data.shape, seed_data.shape))

        logging.info("Loading subject's data.")
        subset, subj_idx = prepare_dataset_for_tracking(hdf5_file, args)

        logging.info("Loading model.")
        model = Learn2TrackModel.load_params_and_state(
            args.experiment_path + '/best_model', log_level=sub_logger_level)
        logging.info("* Formatted model: " +
                     format_dict_to_str(model.params_for_json_prints))

        logging.debug("Instantiating propagator.")
        theta = gm.math.radians(args.theta)
        propagator = RecurrentPropagator(
            subset, subj_idx, model, args.input_group, args.step_size,
            args.rk_order, args.algo, theta, device)

        logging.debug("Instantiating tracker.")
        tracker = DWIMLTracker(
            propagator, mask, seed_generator, nbr_seeds, min_nbr_pts,
            max_nbr_pts, max_invalid_dirs, args.compress, args.nbr_processes,
            args.save_seeds, args.rng_seed, args.track_forward_only,
            use_gpu=args.use_gpu,
            simultanenous_tracking=args.simultaneous_tracking,
            log_level=args.logging)

    return tracker, ref


def _prepare_seed_generator(parser, args, hdf_handle):
    seeding_group = hdf_handle[args.subj_id][args.seeding_mask_group]
    seed_data = np.array(seeding_group['data'], dtype=np.float32)
    seed_res = np.array(seeding_group.attrs['voxres'], dtype=np.float32)

    seed_generator = SeedGenerator(seed_data, seed_res, space=Space.VOX,
                                   origin=Origin('corner'))

    if len(seed_generator.seeds_vox) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    if args.npv:
        # toDo. Not really nb seed per voxel, just in average. Waiting for this
        #  to be modified in scilpy, and we will adapt here.
        nbr_seeds = len(seed_generator.seeds_vox) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds_vox)

    return seed_generator, nbr_seeds, seed_res, seed_data


def _prepare_tracking_mask(args, hdf_handle):
    if args.tracking_mask_group not in hdf_handle[args.subj_id]:
        raise KeyError("HDF group '{}' not found for subject {} in hdf file {}"
                       .format(args.tracking_mask_group, args.subj_id,
                               hdf_handle))
    tm_group = hdf_handle[args.subj_id][args.tracking_mask_group]
    mask_data = np.array(tm_group['data'], dtype=np.float64)
    mask_res = np.array(tm_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(tm_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(mask_data, affine)

    mask = DataVolume(mask_data, mask_res, args.mask_interp)
    return mask, ref, mask_res, mask_data


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

    tracker, ref = prepare_tracker(parser, args, args.hdf5_file, device,
                                   min_nbr_pts, max_nbr_pts, max_invalid_dirs)

    # ----- Track

    with Timer("\nTracking...", newline=True, color='blue'):
        streamlines, seeds = tracker.track()

        logging.debug("Tracked {} streamlines (out of {} seeds). Now saving..."
                      .format(len(streamlines), tracker.nbr_seeds))

    # save seeds if args.save_seeds is given
    data_per_streamline = {'seed': seeds} if args.save_seeds else {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    sft = StatefulTractogram(streamlines, ref, Space.VOXMM,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram,
                    bbox_valid_check=False)


if __name__ == "__main__":
    main()
