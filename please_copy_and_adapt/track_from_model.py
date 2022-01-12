#! /usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script allows tracking from a trained Learn2track model.

See scilpy's scil_compute_local_tracking and scil_compute_local_tracking_2
for references. However, this version does not use deterministic nor
probabilistic tracking per say. Propagation (choosing the next direction)
depends on the output of the model.
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
                             verify_compression_th, add_processes_arg)
from scilpy.tracking.utils import (add_seeding_options,
                                   verify_streamline_length_options,
                                   verify_seed_options, add_out_options)

from dwi_ml.data.dataset.single_subject_containers import SubjectData
from dwi_ml.data.dataset.mri_data_containers import MRIData
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.tracking.seed import DWIMLSeedGenerator
from dwi_ml.tracking.utils import (add_mandatory_options_tracking,
                                   add_tracking_options)

##################
# PLEASE COPY AND ADAPT:
##################
# Use your own model.
from dwi_ml.models.main_models import MainModelWithPD

# Choose appropriate classes or implement your own child classes.
# Example:
from dwi_ml.tracking.tracker import DWIMLTracker
from dwi_ml.tracking.propagator import DWIMLPropagatorOneInputAndPD


def build_argparser():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    add_mandatory_options_tracking(p)

    track_g = add_tracking_options(p)
    # Sphere used if the direction_getter key is the sphere-classification.
    add_sphere_arg(track_g, symmetric_only=False)

    add_seeding_options(p)

    r_g = p.add_argument_group('  Random seeding options')
    r_g.add_argument('--rng_seed', type=int,
                     help='Initial value for the random number generator. '
                          '[%(default)s]')
    r_g.add_argument('--skip', type=int, default=0,
                     help="Skip the first N random number. \n"
                          "Useful if you want to create new streamlines to "
                          "add to \na previously created tractogram with a "
                          "fixed --rng_seed.\nEx: If tractogram_1 was created "
                          "with -nt 1,000,000, \nyou can create tractogram_2 "
                          "with \n--skip 1,000,000.")

    m_g = p.add_argument_group('  Memory options')
    ram_options = m_g.add_mutually_exclusive_group()
    # Parallel processing or GPU processing
    add_processes_arg(ram_options)
    ram_options.add_argument('--use_gpu', action='store_true',
                             help="If set, use GPU for processing. Cannot be "
                                  "used \ntogether with --processes.")

    # toDo. This should be clarified in scilpy eventually. Verifiy evolution.
    m_g.add_argument('--set_mmap_to_none', action='store_true',
                     help="If true, use mmap_mode=None. Else mmap_mode='r+'.\n"
                          "Used in np.load(data_file_info). Available only "
                          "with --processes.\nTO BE CLEANED")
    add_out_options(p)

    p.add_argument('--logging', metavar='level',
                   choices=['info', 'debug', 'warning'], default='warning',
                   help="Logging level. One of 'debug', 'info' or 'warning'.")

    return p


def perform_checks(parser, args):
    """
    Verifying options
    """
    if not nib.streamlines.is_supported(args.out_tractogram):
        parser.error('Invalid output streamline file format (must be trk or ' +
                     'tck): {0}'.format(args.out_tractogram))

    if not (args.sm_from_hdf5 or args.sm_from_data):
        parser.error("One seeding mask value must be given amongst "
                     "--sm_from_hdf5 or --sm_from_data.")
    if not (args.tm_from_hdf5 or args.tm_from_data):
        parser.error("One tracking mask value must be given amongst "
                     "--tm_from_hdf5 or --tm_from_data.")
    if not (args.input_from_hdf5 or args.input_from_data):
        parser.error("One input value must be given amongst "
                     "--input_from_hdf5 or --input_from_data.")

    hdf5_needed = any([args.sm_from_hdf5, args.tm_from_hdf5,
                       args.input_from_hdf5])
    if hdf5_needed:
        if args.hdf5_file is None:
            parser.error("hdf5 file must be given!")
        assert_inputs_exist(parser, args.hdf5_file)

    if args.sm_from_data:
        assert_inputs_exist(parser, args.sm_from_data)
    if args.tm_from_data:
        assert_inputs_exist(parser, args.tm_from_data)
    if args.input_from_data:
        assert_inputs_exist(parser, args.input_from_data)

    assert_outputs_exist(parser, args, [args.out_tractogram,
                                        args.experiment_path,
                                        args.experiment_path + '/model'])

    verify_streamline_length_options(parser, args)
    verify_compression_th(args.compress)
    verify_seed_options(parser, args)

    return hdf5_needed


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
        subj_data, volume_group = _prepare_data(parser, args, hdf_handle)

        logging.info("Loading model.")
        model = MainModelWithPD.load(args.experiment_path + '/model')
        logging.info("* Loaded params: " + format_dict_to_str(model.params) +
                     "\n")
        logging.info("* Formatted model: " +
                     format_dict_to_str(model.params_per_layer))

        logging.debug("Instantiating propagator.")
        theta = gm.math.radians(args.theta)
        propagator = DWIMLPropagatorOneInputAndPD(
            subj_data, model, volume_group, model.neighborhood_points,
            args.step_size, args.rk_order, args.algo, theta, device)

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
    if args.sm_from_hdf5:
        seed_data = np.array(
            hdf_handle[args.subj_id][args.sm_from_hdf5]['data'],
            dtype=np.float32)
        seed_res = np.array(
            hdf_handle[args.subj_id][args.sm_from_hdf5].attrs['voxres'],
            dtype=np.float32)
    else:
        seed_img = nib.load(args.sm_from_data)
        seed_data = seed_img.get_fdata(dtype=float)
        seed_res = seed_img.header.get_zooms()[:3]
    seed_generator = DWIMLSeedGenerator(seed_data, seed_res, device)

    if len(seed_generator.seeds) == 0:
        parser.error('Seed mask "{}" does not have any voxel with value > 0.'
                     .format(args.in_seed))

    if args.npv:
        # toDo. Not really nb seed per voxel, just in average.
        nbr_seeds = len(seed_generator.seeds) * args.npv
    elif args.nt:
        nbr_seeds = args.nt
    else:
        # Setting npv = 1.
        nbr_seeds = len(seed_generator.seeds)

    return seed_generator, nbr_seeds


def _prepare_tracking_mask(args, hdf_handle):
    if args.tm_from_hdf5:
        mask_data = np.array(
            hdf_handle[args.subj_id][args.tm_from_hdf5]['data'],
            dtype=np.float64)
        mask_res = np.array(
            hdf_handle[args.subj_id][args.tm_from_hdf5].attrs['voxres'],
            dtype=np.float32)
        affine = np.array(
            hdf_handle[args.subj_id][args.tm_from_hdf5].attrs['affine'],
            dtype=np.float32)
        ref = nib.Nifti1Image(mask_data, affine)
    else:
        mask_img = nib.load(args.tm_from_data)
        mask_data = mask_img.get_fdata(dtype=float)
        mask_res = mask_img.header.get_zooms()[:3]
        ref = mask_img

    mask = DataVolume(mask_data, mask_res, args.mask_interp)
    return mask, ref


def _prepare_data(parser, args, hdf_handle):
    if args.input_from_hdf5:
        if not args.subj_id:
            parser.error("Subject id must be given to retrieve data from "
                         "hdf5.")
        subj_data = SubjectData.init_from_hdf(
            args.subj_id, logging.getLogger(), hdf_handle,
            group_info=None)
        volume_group = args.input_from_hdf5
    else:
        volume_img = nib.load(args.input)
        volume_data = volume_img.get_fdata(dtype=float)
        volume_res = volume_img.header.get_zooms()[:3]
        mri_data = MRIData(volume_data, volume_res,
                           interpolation=args.data_interp)
        nb_features = volume_img.shape[-1]
        subj_data = SubjectData(subject_id='subj_for_tracking',
                                volume_groups=['in_data'],
                                nb_features=[nb_features],
                                mri_data_list=[mri_data],
                                streamline_groups=None, sft_data_list=None)
        volume_group = 'in_data'

    return subj_data, volume_group


def main():
    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=args.logging.upper())

    hdf5_needed = perform_checks(parser, args)

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

    hdf_handle = None
    if hdf5_needed:
        hdf_handle = h5py.File(args.hdf5_file, 'r')

    tracker, ref = prepare_tracker(parser, args, hdf_handle, device,
                                   min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                                   mmap_mode)

    with Timer("\n\nTracking...",
               newline=True, color='blue'):
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
    save_tractogram(sft, args.out_tractogram)


if __name__ == "__main__":
    main()
