#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Learn2track model.
"""
import argparse
import logging
import math

import dipy.core.geometry as gm
from dipy.io.stateful_tractogram import (StatefulTractogram, Space,
                                         set_sft_logger_level)
from dipy.io.streamline import save_tractogram
import h5py
import nibabel as nib
import numpy as np
import torch

from scilpy.image.datasets import DataVolume
from scilpy.io.utils import (add_sphere_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             verify_compression_th)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.data.dataset.single_subject_containers import SubjectData
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInput
from dwi_ml.tracking.seed import DWIMLSeedGenerator
from dwi_ml.tracking.tracker import DWIMLTracker
from dwi_ml.tracking.utils import (add_mandatory_options_tracking,
                                   add_tracking_options)

from TransformingTractography.models.transformer import \
    OriginalTransformerModel


def build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    # Sphere used if the direction_getter key is the sphere-classification.
    add_sphere_arg(track_g, symmetric_only=False)

    # As in scilpy:
    add_seeding_options(p)
    add_out_options(p)

    p.add_argument('--logging', metavar='level',
                   choices=['info', 'debug', 'warning'], default='info',
                   help="Logging level. One of 'debug', 'info' or 'warning'.")

    return p


def prepare_tracker(parser, args, hdf_handle, device,
                    min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                    mmap_mode):
    with Timer("\n\nPreparing everything...",
               newline=True, color='cyan'):
        logging.info("Loading seeding mask + preparing seed generator.")
        seed_generator, nbr_seeds = _prepare_seed_generator(parser, args,
                                                            hdf_handle, device)

        logging.info("Loading tracking mask.")
        mask, ref = _prepare_tracking_mask(args, hdf_handle)

        logging.info("Loading subject's data.")
        subj_data = SubjectData.init_from_hdf(args.subj_id, hdf_handle,
                                              group_info=None)

        logging.info("Loading model.")
        model = OriginalTransformerModel.load(args.experiment_path + '/model')
        model.set_logger_state(args.logging.upper())
        logging.info("* Loaded params: " + format_dict_to_str(model.params)
                     + "\n")

        logging.debug("Instantiating propagator.")
        theta = gm.math.radians(args.theta)
        propagator = DWIMLPropagatorOneInput(
            subj_data, model, args.input_group, args.step_size, args.rk_order,
            args.algo, theta, device)

        logging.debug("Instantiating tracker.")
        if args.nbr_processes > 1:
            # toDo
            raise NotImplementedError(
                "Usage with --processes>1 not ready in dwi_ml! "
                "See the #toDo in scilpy! It uses tracking_field.dataset.data "
                "which does not exist in our case!")
        tracker = DWIMLTracker(
            propagator, mask, seed_generator, nbr_seeds, min_nbr_pts,
            max_nbr_pts, max_invalid_dirs, args.compress, args.nbr_processes,
            args.save_seeds, mmap_mode, args.rng_seed, args.track_forward_only,
            args.use_gpu)

    return tracker, ref


def _prepare_seed_generator(parser, args, hdf_handle, device):
    seeding_group = hdf_handle[args.subj_id][args.seeding_mask_group]
    seed_data = np.array(seeding_group['data'], dtype=np.float32)
    seed_res = np.array(seeding_group.attrs['voxres'], dtype=np.float32)

    seed_generator = DWIMLSeedGenerator(seed_data, seed_res, device)

    if len(seed_generator.seeds) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    if args.npv:
        # toDo. Not really nb seed per voxel, just in average. Waiting for this
        #  to be modified in scilpy, and we will adapt here.
        nbr_seeds = len(seed_generator.seeds) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds)

    return seed_generator, nbr_seeds


def _prepare_tracking_mask(args, hdf_handle):
    tm_group = hdf_handle[args.subj_id][args.tracking_mask_group]
    mask_data = np.array(tm_group['data'], dtype=np.float64)
    mask_res = np.array(tm_group.attrs['voxres'], dtype=np.float32)
    affine = np.array(tm_group.attrs['affine'], dtype=np.float32)
    ref = nib.Nifti1Image(mask_data, affine)

    mask = DataVolume(mask_data, mask_res, args.mask_interp)
    return mask, ref


def main():
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=args.logging.upper())

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

    # r+ is necessary for interpolation function in cython who need read/write
    # rights
    mmap_mode = None if args.set_mmap_to_none else 'r+'

    device = torch.device('cpu')
    if args.use_gpu:
        if args.nbr_processes > 1:
            logging.warning("Number of processes was set to {} but you "
                            "are using GPU. Parameter ignored."
                            .format(args.nbr_processes))
        if torch.cuda.is_available():
            device = torch.device('cuda')

    hdf_handle = h5py.File(args.hdf5_file, 'r')

    tracker, ref = prepare_tracker(parser, args, hdf_handle, device,
                                   min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                                   mmap_mode)

    # ----- Track

    with Timer("\nTracking...", newline=True, color='blue'):
        streamlines, seeds = tracker.track()

        logging.debug("Tracked {} streamlines (out of {} seeds). Now saving..."
                      .format(len(streamlines), tracker.nbr_seeds))

    # save seeds if args.save_seeds is given
    data_per_streamline = {'seed': lambda: seeds} if args.save_seeds else {}

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    sft = StatefulTractogram(streamlines, ref, Space.VOXMM,
                             data_per_streamline=data_per_streamline)
    save_tractogram(sft, args.out_tractogram,
                    bbox_valid_check=False)


if __name__ == "__main__":
    main()
