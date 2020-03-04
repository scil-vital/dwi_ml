#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import datetime
import json
import logging
import shutil
from pathlib import Path
from typing import Dict

import h5py
import nibabel as nib

from dwi_ml.data.creation.create_hdf5_utils import (
    verify_subject_lists,
    process_group,
    process_streamlines)

DESCRIPTION = """Script to combine multiple diffusion MRI volumes and their 
streamlines into a single .hdf5 file.

Your folders should follow organization described in 
dwi_ml.data.creation.description_data_structure.FOLDER_DESCRIPTION.

A hdf5 folder will be created alongside dwi_ml_ready. It will contain the .hdf5
file and possibly intermediate files created during bundle processing.
"""

def _parse_args():
    p = argparse.ArgumentParser(description=DESCRIPTION,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('path', type=str,
                   help="Path to database folder, which should contain a "
                        "dwi_ml_ready folder.")
    p.add_argument('config_file', type=str,
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. Should follow description in "
                        "dwi_ml.data.creation.description_data_structure's "
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
    p.add_argument('--name', type=str,
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--mask', type=str,
                   help="Mask defining the voxels used for data "
                        "standardization. If none is given, all non-zero "
                        "voxels will be used. mask should be the name of "
                        "the mask file inside dwi_ml_ready/{subj_id}/mask.")
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
                      args.mask, save_intermediate=args.save_intermediate,
                      force=args.force)

def _generate_dataset(path: str, name: str, chosen_subjs, groups_config: Dict,
                      mask: str, save_intermediate: bool = False,
                      force: bool = False):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.

    Parameters
    ----------
    path : str
        Path to dataset folder, which contains dwi_ml_ready.
    name : str
        Dataset name used to save the .hdf5 file. If not given, guessed from
        path name.
    dataset_creator : DatasetCreator
        Configuration for the dataset to generate.
    groups_config: Dict
        List of groups and associated inputs from the json file.
    mask: str or None
        Name of the mask for each subject (in their dwi_ml mask folder).
    save_intermediate : bool
        Save intermediate processing files for each subject.
    force : bool
        Overwrite an existing dataset if it exists.
    """
    # Create hdf5 folder or clean existing one
    hdf5_path = Path(path, "hdf5")
    if hdf5_path.is_dir():
        if force:
            print("Deleting existing hdf5 data folder: {}"
                  .format(hdf5_path))
            shutil.rmtree(str(hdf5_path))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. "
                                  "Use force to allow overwrite"
                                  .format(hdf5_path))
    hdf5_path.mkdir()

    # Initialize database
    if name:
        dataset_name = name
    else:
        dataset_name = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))
    dataset_file = hdf5_path.joinpath("{}.hdf5".format(dataset_name))

    # Combine data into a hdf file.
    dwi_ml_folder = Path(path, "dwi_ml_ready")
    with h5py.File(dataset_file, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        hdf_file.attrs['chosen_subjs'] = chosen_subjs
        hdf_file.attrs['groups'] = groups_config
        hdf_file.attrs['step_size_mm'] = step_size
        hdf_file.attrs['bundles'] = [b.name for b in bundles]

        # Starting the subjects processing
        logging.info("Processing {} subjects : {}".format(len(chosen_subjs),
                                                          chosen_subjs))
        for subj_id in chosen_subjs:
            logging.info("Processing subject: {}".format(subj_id))
            subj_input_path = dwi_ml_folder.joinpath(subj_id)

            # Add subject to hdf database
            subj_hdf = hdf_file.create_group(subj_id)

            # Create subject's hdf5 data folder
            subj_output_path = hdf5_path.joinpath(subj_id)
            subj_output_path.mkdir()

            # Find subject's mask
            if mask is not None:
                subj_mask_file = subj_input_path.joinpath('masks/' + mask)
                subj_mask_data = nib.load(subj_mask_file).get_fdata(type=bool)

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                file_list = groups_config[group]
                group_data = process_group(group, file_list, save_intermediate,
                                           subj_input_path, subj_output_path,
                                           subj_mask_data)
                subj_hdf.create_dataset(group, data=group_data)

            # Add the streamlines data
            streamlines, lengths = process_streamlines(subj_id, subj_input_path,
                                 subj_output_path, dataset_creator,
                                 save_intermediate=save_intermediate)
            if streamlines is not None:
                streamlines_group = AJOUTER LES SPACE ATTRIBUTES
                streamlines_group = subj_hdf.create_group('streamlines')
                # Accessing private Dipy values, but necessary
                streamlines_group.create_dataset('data',
                                                 data=streamlines._data)
                streamlines_group.create_dataset('offsets',
                                                 data=streamlines._offsets)
                streamlines_group.create_dataset('lengths',
                                                 data=streamlines._lengths)
                streamlines_group.create_dataset('euclidean_lengths',
                                                 data=lengths)

    print("Saved dataset : {}".format(dataset_file))


if __name__ == '__main__':
    main()
