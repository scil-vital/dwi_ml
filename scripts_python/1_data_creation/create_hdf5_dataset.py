#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to combine multiple diffusion MRI volumes and their streamlines into a
single .hdf5 file.

Your folders should follow organization described in
dwi_ml.data.hdf5_creation.description_data_structure.FOLDER_DESCRIPTION.

A hdf5 folder will be created alongside dwi_ml_ready. It will contain the .hdf5
file and possibly intermediate files created during bundle processing.
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

from dwi_ml.data.hdf5_creation.utils import (verify_subject_lists,
                                             process_group,
                                             process_streamlines)


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path', type=str,
                   help="Path to database folder, which should contain a "
                        "dwi_ml_ready folder.")
    p.add_argument('config_file', type=str,
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. Should follow description in "
                        "dwi_ml.data.hdf5_creation.description_data_structure's "
                        "CONFIG_DESCRIPTION")
    p.add_argument('dwi', type=str,
                   help="Name of the DWI data that should be found inside "
                        "each subject's dwi folder in dwi_ml_ready.")
    p.add_argument('training_subj_ids', type=str,
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subj_ids', type=str,
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('--step_size', type=float,
                   help="Common step size to resample the data. Must be in the "
                        "same space as the streamlines. If none is given, we "
                        "will compress the streamlines instead of resampling. ")
    p.add_argument('--name', type=str,
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--bundles', type=str, nargs='+',
                   help="Bundles to concatenate as streamlines in the hdf5. "
                        "Must be names of files that are present in each "
                        "subject's 'bundles' folder in dwi_ml_ready. If none "
                        "is given, we will take all the files in the 'bundles'"
                        "folder. The bundles names will be guessed from the "
                        "filenames.")
    p.add_argument('--mask', type=str,
                   help="Mask defining the voxels used for data "
                        "standardization. If none is given, all non-zero "
                        "voxels will be used. mask should be the name of "
                        "the mask file inside dwi_ml_ready/{subj_id}/mask.")
    p.add_argument('--space', type=Space, default=Space.VOX,
                   help="Default space to bring all the stateful tractograms. "
                        "All other measures (ex, step_size) should be provided "
                        "in that space.")
    p.add_argument('--logging', type=str,
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

    # Create dataset from config and save
    _generate_dataset(args.path, args.name, chosen_subjs, groups_config,
                      args.mask, args.bundles, args.step_size, args.space,
                      save_intermediate=args.save_intermediate,
                      force=args.force)


def _generate_dataset(database_dir: str, name: str, chosen_subjs: List[str],
                      groups_config: Dict, mask: str, bundles: List[str],
                      step_size: float, space: Space,
                      save_intermediate: bool = False,
                      force: bool = False):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.

    Parameters
    ----------
    database_dir : str
        Path to dataset folder, which contains dwi_ml_ready.
    name : str
        Dataset name used to save the .hdf5 file. If not given, guessed from
        path name.
    chosen_subjs: List[str]
        The list of subject ids to be included.
    groups_config: Dict
        List of groups and associated inputs from the json file.
    mask: str or None
        Name of the mask for each subject (in their dwi_ml mask folder).
    bundles: List[str]
        The list of bundles to be included for each subject.
    step_size: float
        Step size to resample the streamlines.
    space: Space
        Space to consider the streamlines.
    save_intermediate : bool
        Save intermediate processing files for each subject.
    force : bool
        Overwrite an existing dataset if it exists.
    """
    dwi_ml_dir = Path(database_dir, "dwi_ml_ready")

    # Initialize database
    hdf5_dir, hdf5_file_path = initialize_hdf5_database(database_dir,
                                                        force, name)

    # Add data to database
    with h5py.File(hdf5_file_path, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        hdf_file.attrs['chosen_subjs'] = chosen_subjs
        hdf_file.attrs['groups'] = groups_config
        hdf_file.attrs['step_size'] = step_size
        hdf_file.attrs['bundles'] = [b for b in bundles]
        hdf_file.attrs['space'] = space

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
            if save_intermediate:
                subj_intermediate_path.mkdir()

            # Find subject's mask
            if mask is not None:
                subj_mask_file = subj_input_dir.joinpath('masks/' + mask)
                subj_mask_data = nib.load(subj_mask_file).get_fdata(type=bool)

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                file_list = groups_config[group]
                group_data, group_affine, group_header = process_group(
                    group, file_list, save_intermediate, subj_input_dir,
                    subj_intermediate_path, subj_mask_data)
                subj_hdf.create_dataset(group, data=group_data)

            # Taking any group (the last) to save space information.
            # All groups should technically have the same information.
            subj_hdf.attrs['affine'] = group_affine
            subj_hdf.attrs['header'] = group_header

            # Add the streamlines data
            tractogram, lengths = process_streamlines(
                subj_input_dir.joinpath("bundles"), bundles, group_header,
                step_size, space)
            streamlines = tractogram.streamlines

            # Save streamlines
            if save_intermediate:
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

    # Initialize database
    if name:
        hdf5_name = name
    else:
        hdf5_name = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    hdf5_file_path = hdf5_dir.joinpath("{}.hdf5".format(hdf5_name))

    return hdf5_dir, hdf5_file_path


if __name__ == '__main__':
    main()
