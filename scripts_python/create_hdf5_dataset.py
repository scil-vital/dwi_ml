#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to combine multiple diffusion MRI volumes and their streamlines into a
single .hdf5 file.

You should have a file dwi_ml_ready organized as prescribed (see our doc for
more information.) A hdf5 folder will be created alongside dwi_ml_ready.
It will contain the .hdf5 file and possibly intermediate files.
"""

import argparse
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List

import h5py
import nibabel as nib
import numpy as np
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space

from dwi_ml.data.hdf5_creation import (process_group, process_streamlines,
                                       verify_subject_lists)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('database_folder',
                   help="Path to database folder, which should contain a "
                        "dwi_ml_ready folder.")
    p.add_argument('config_file',
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. Should follow description in "
                        "our doc.")  # toDo. Add link when created.
    p.add_argument('training_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('--step_size', type=float,
                   help="Common step size to resample the data. Must be in the "
                        "same space as the streamlines.\nIf none is given, we "
                        "will compress the streamlines instead of resampling. ")
    p.add_argument('--name',
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--bundles', nargs='+',
                   help="Bundles to concatenate as streamlines in the hdf5. "
                        "The bundles names will be guessed from the "
                        "filenames.\nMust be names of files that are present "
                        "in each subject's 'bundles' folder in dwi_ml_ready.\n"
                        "If none is given, we will take all the files in the "
                        "'bundles' folder.")
    p.add_argument('--mask',
                   help="Mask defining the voxels used for data "
                        "standardization.\nIf none is given, all non-zero "
                        "voxels will be used. Should be the name of "
                        "a file inside dwi_ml_ready/{subj_id}.")
    p.add_argument('--space', type=str, default=Space.VOX,
                   choices=['Space.RASMM', 'Space.VOX', 'Space.VOXMM'],
                   help="Default space to bring all the stateful tractograms. "
                        "All other measures (ex, step_size) should be provided "
                        "in that space.")
    p.add_argument('--logging',
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Choose logging level. [warning]")
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject inside")
    p.add_argument('-f', '--force', action='store_true',
                   help="If set, overwrite existing hdf5 folder.")

    arguments = p.parse_args()

    return arguments


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    args = _parse_args()

    # Initialize logger
    logging.basicConfig(level=str(args.logging).upper())

    # Verify that dwi_ml_ready folder is found
    dwi_ml_ready_folder = Path(args.database_folder + '/dwi_ml_ready')
    if not dwi_ml_ready_folder.is_dir():
        raise ValueError('The dwi_ml_ready folder was not found: '
                         '{}'.format(dwi_ml_ready_folder))

    # Read training subjs and validation subjs
    with open(args.training_subjs, 'r') as file:
        training_subjs = file.read().splitlines()
    with open(args.validation_subjs, 'r') as file:
        validation_subjs = file.read().splitlines()

    # Verify that subjects exist and that no subjects are forgotten
    chosen_subjs = training_subjs + validation_subjs
    verify_subject_lists(dwi_ml_ready_folder, chosen_subjs)

    # Read group information from the json file
    json_file = open(args.config_file, 'r')
    groups_config = json.load(json_file)

    # Check if hdf5 already exists and initialize database
    hdf5_dir, hdf5_filename = _initialize_hdf5_database(args.database_folder,
                                                        args.force, args.name)

    # Create dataset from config and save
    _add_all_subjs_to_database(args, chosen_subjs, groups_config, hdf5_dir,
                               hdf5_filename, dwi_ml_ready_folder)


def _initialize_hdf5_database(database_path, force, name):
    # Create hdf5 dir or clean existing one
    hdf5_dir = Path(database_path, "hdf5")
    if hdf5_dir.is_dir():
        if force:
            logging.info("Careful! Deleting existing hdf5 data folder: "
                         "{}".format(hdf5_dir))
            shutil.rmtree(str(hdf5_dir))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. Use "
                                  "force to allow overwrite.".format(hdf5_dir))
    hdf5_dir.mkdir()

    # Define dabase name
    if name:
        hdf5_filename = name
    else:
        hdf5_filename = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))

    return hdf5_dir, hdf5_filename


def _add_all_subjs_to_database(args, chosen_subjs: List[str],
                               groups_config: Dict, hdf5_dir, hdf5_filename,
                               dwi_ml_dir):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.
    """
    # Cast space from str to Space
    if args.space == 'Space.Vox':
        space = Space.Vox
    elif args.space == 'Space.RASMM':
        space = Space.RASMM
    else:
        space = Space.VOXMM

    # Add data to database
    hdf5_file_path = hdf5_dir.joinpath("{}.hdf5".format(hdf5_filename))
    with h5py.File(hdf5_file_path, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        now = datetime.datetime.now()
        hdf_file.attrs['data_and_time'] = now.strftime('%d %B %Y %X')
        hdf_file.attrs['chosen_subjs'] = chosen_subjs
        hdf_file.attrs['groups'] = str(groups_config)
        if args.step_size is not None:
            hdf_file.attrs['step_size'] = args.step_size
        else:
            hdf_file.attrs['step_size'] = 'Not defined by user'
        hdf_file.attrs['bundles'] = [b for b in args.bundles]
        hdf_file.attrs['space'] = args.space

        # Add data one subject at the time
        logging.info("Processing {} subjects : "
                     "{}".format(len(chosen_subjs), chosen_subjs))
        for subj_id in chosen_subjs:
            logging.info("\n *Processing subject: {}".format(subj_id))
            subj_input_dir = dwi_ml_dir.joinpath(subj_id)

            # Add subject to hdf database
            subj_hdf = hdf_file.create_group(subj_id)

            # Prepare subject folder for intermediate files
            subj_intermediate_path = hdf5_dir.joinpath(subj_id)
            if args.save_intermediate:
                subj_intermediate_path.mkdir()

            # Find subject's mask
            if args.mask:
                subj_mask_file = subj_input_dir.joinpath(args.mask)
                subj_mask_img = nib.load(subj_mask_file)
                subj_mask_data = np.asanyarray(subj_mask_img.dataobj)
                subj_mask_data = subj_mask_data.astype(np.bool)

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                logging.debug('Processing group {}...'.format(group))
                file_list = groups_config[group]
                group_data, group_affine, group_header = process_group(
                    group, file_list, args.save_intermediate, subj_input_dir,
                    subj_intermediate_path, subj_mask_data)
                logging.debug('...done. Now creating dataset from group.')
                subj_hdf.create_dataset(group, data=group_data)
                logging.debug('...done.')

            # Reparing header. Should be arranged in Dipy 1.2.
            group_header['srow_x'] = group_affine[0, :]
            group_header['srow_y'] = group_affine[1, :]
            group_header['srow_z'] = group_affine[2, :]

            # Taking any group (the last) to save space information.
            # All groups should technically have the same information.
            subj_hdf.attrs['affine'] = group_affine
            subj_hdf.attrs['header'] = str(group_header)

            # Add the streamlines data
            logging.debug('Processing bundles...')
            tractogram, lengths = process_streamlines(
                subj_input_dir.joinpath("bundles"), args.bundles, group_header,
                args.step_size, space)
            streamlines = tractogram.streamlines

            # Save streamlines
            if args.save_intermediate:
                logging.debug('Saving intermediate tractogram.')
                save_tractogram(tractogram,
                                str(subj_intermediate_path.joinpath(
                                    "{}_all_streamlines.tck".format(subj_id))))

            if streamlines is None:
                logging.warning('Careful! Total tractogram for subject {} '
                                'contained no streamlines!'.format(subj_id))
            else:
                streamlines_group = subj_hdf.create_group('streamlines')

                streamlines_group.attrs['space_attributes'] = \
                    str(tractogram.space_attributes)
                streamlines_group.attrs['space'] = space

                # Accessing private Dipy values, but necessary
                streamlines_group.create_dataset('data',
                                                 data=streamlines._data)
                streamlines_group.create_dataset('offsets',
                                                 data=streamlines._offsets)
                streamlines_group.create_dataset('lengths',
                                                 data=streamlines._lengths)
                streamlines_group.create_dataset('euclidean_lengths',
                                                 data=lengths)

    print("Saved dataset : {}".format(hdf5_file_path))


if __name__ == '__main__':
    main()
