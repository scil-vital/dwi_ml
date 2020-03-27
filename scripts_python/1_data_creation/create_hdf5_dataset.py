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
from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space

from dwi_ml.data.hdf5_creation import (process_group, process_streamlines,
                                       verify_subject_lists)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path',
                   help="Path to database folder, which should contain a "
                        "dwi_ml_ready folder.")
    p.add_argument('config_file',
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. Should follow description in "
                        "our doc.")  # toDo. Add link when created.
    p.add_argument('training_subj_ids',
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subj_ids',
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('--step_size', type=float,
                   help="Common step size to resample the data. Must be in the "
                        "same space as the streamlines. If none is given, we "
                        "will compress the streamlines instead of resampling. ")
    p.add_argument('--name',
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--bundles', nargs='+',
                   help="Bundles to concatenate as streamlines in the hdf5. "
                        "Must be names of files that are present in each "
                        "subject's 'bundles' folder in dwi_ml_ready. If none "
                        "is given, we will take all the files in the 'bundles'"
                        "folder. The bundles names will be guessed from the "
                        "filenames.")
    p.add_argument('--mask',
                   help="Mask defining the voxels used for data "
                        "standardization. If none is given, all non-zero "
                        "voxels will be used. mask should be the name of "
                        "the mask file inside dwi_ml_ready/{subj_id}/mask.")
    p.add_argument('--space', type=Space, default=Space.VOX,
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
    logging.info(args)

    dwi_ml_folder = Path(args.path, "dwi_ml_ready")

    # Verify that subjects exist and that no subjects are forgotten
    chosen_subjs = args.training_subj_ids + args.validation_subj_ids
    verify_subject_lists(dwi_ml_folder, chosen_subjs)

    # Read group information from the json file
    json_file = open(args.config_file, 'r')
    groups_config = json.load(json_file)

    dwi_ml_dir = Path(args.path, "dwi_ml_ready")

    # Initialize database
    hdf5_dir, hdf5_filename = initialize_hdf5_database(args.path, args.force,
                                                       args.name)

    # Create dataset from config and save
    add_all_subjs_to_database(args, chosen_subjs, groups_config, hdf5_dir,
                              hdf5_filename, dwi_ml_dir)


def initialize_hdf5_database(database_path, force, name):
    # Create hdf5 dir or clean existing one
    hdf5_dir = Path(database_path, "hdf5")
    if hdf5_dir.is_dir():
        if force:
            print("Deleting existing hdf5 data folder: {}"
                  .format(hdf5_dir))
            shutil.rmtree(str(hdf5_dir))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. "
                                  "Use force to allow overwrite"
                                  .format(hdf5_dir))
    hdf5_dir.mkdir()

    # Define dabase name
    if name:
        hdf5_filename = name
    else:
        hdf5_filename = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))

    return hdf5_dir, hdf5_filename


def add_all_subjs_to_database(args, chosen_subjs: List[str],
                              groups_config: Dict, hdf5_dir, hdf5_filename,
                              dwi_ml_dir):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.
    """

    # Add data to database
    hdf5_file_path = hdf5_dir.joinpath("{}.hdf5".format(hdf5_filename))
    with h5py.File(hdf5_file_path, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        hdf_file.attrs['chosen_subjs'] = chosen_subjs
        hdf_file.attrs['groups'] = groups_config
        hdf_file.attrs['step_size'] = args.step_size
        hdf_file.attrs['bundles'] = [b for b in args.bundles]
        hdf_file.attrs['space'] = args.space

        # Add data one subject at the time
        logging.info("Processing {} subjects : "
                     "{}".format(len(chosen_subjs), chosen_subjs))
        for subj_id in chosen_subjs:
            logging.info("Processing subject: {}".format(subj_id))
            subj_input_dir = dwi_ml_dir.joinpath(subj_id)

            # Add subject to hdf database
            subj_hdf = hdf_file.create_group(subj_id)

            # Prepare subject folder for intermediate files
            subj_intermediate_path = hdf5_dir.joinpath(subj_id)
            if args.save_intermediate:
                subj_intermediate_path.mkdir()

            # Find subject's mask
            if args.mask:
                subj_mask_file = subj_input_dir.joinpath('masks/' + args.mask)
                subj_mask_data = nib.load(subj_mask_file).get_fdata(type=bool)

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                file_list = groups_config[group]
                group_data, group_affine, group_header = process_group(
                    group, file_list, args.save_intermediate, subj_input_dir,
                    subj_intermediate_path, subj_mask_data)
                subj_hdf.create_dataset(group, data=group_data)

            # Taking any group (the last) to save space information.
            # All groups should technically have the same information.
            subj_hdf.attrs['affine'] = group_affine
            subj_hdf.attrs['header'] = group_header

            # Add the streamlines data
            tractogram, lengths = process_streamlines(
                subj_input_dir.joinpath("bundles"), args.bundles, group_header,
                args.step_size, args.space)
            streamlines = tractogram.streamlines

            # Save streamlines
            if args.save_intermediate:
                save_tractogram(tractogram,
                                str(subj_intermediate_path.joinpath(
                                    "{}_all_streamlines.tck".format(subj_id))))

            if streamlines is None:
                logging.info('Streamlines not added to hdf5 for subj {}. '
                             'Bundle did not exist?'.format(subj_id))
            else:
                streamlines_group = subj_hdf.create_group('streamlines')

                streamlines_group.attrs['space_attributes'] = \
                    tractogram.space_attributes
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
