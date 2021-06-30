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
import glob
import logging
import shutil
from pathlib import Path
from typing import Dict, List

from dipy.io.streamline import save_tractogram
from dipy.io.stateful_tractogram import Space
import h5py
import nibabel as nib
import numpy as np
from nested_lookup import nested_lookup

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
                        "our doc, here: "
                        "https://dwi-ml.readthedocs.io/en/latest/data_organization.html")
    p.add_argument('training_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('--step_size', type=float,
                   help="Common step size to resample the data. Must be in "
                        "the same space as the streamlines.\nIf none is "
                        "given, we will compress the streamlines instead of "
                        "resampling.")
    p.add_argument('--name',
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--bundles', nargs='+',
                   help="Bundles to concatenate as streamlines in the hdf5. "
                        "Must correspond to the filenames present "
                        "in each subject's 'bundles' folder in dwi_ml_ready.\n"
                        "If none is given, we will take all the files in the "
                        "'bundles' folder.")
    p.add_argument('--enforce_bundles_presence', type=bool, default=True,
                   help='If true, the process will stop if one bundle is '
                        'missing for a subject.')
    p.add_argument('--std_mask',
                   help="Mask defining the voxels used for data "
                        "standardization. Should be the name of a file inside "
                        "dwi_ml_ready/{subj_id}\nStandardization is performed "
                        "on each channel separatedly.\nIf none is given, all "
                        "non-zero voxels will be used.")
    p.add_argument('--space', type=str, default='vox',
                   choices=['rasmm', 'vox', 'voxmm'],
                   help="Default space to bring all the stateful tractograms. "
                        "All other measures (ex, step_size) should be "
                        "provided in that space.")
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
        raise argparse.ArgumentError('The dwi_ml_ready folder was not found: '
                                     '{}'.format(dwi_ml_ready_folder))
    else:
        logging.info('***Creating hdf5 dataset')
        logging.debug('   Creating hdf5 data from data in the following foler:'
                      ' {}'.format(dwi_ml_ready_folder))

    # Read training subjs and validation subjs
    with open(args.training_subjs, 'r') as file:
        training_subjs = file.read().split()
        logging.debug('   Training subjs: {}'.format(training_subjs))
    with open(args.validation_subjs, 'r') as file:
        validation_subjs = file.read().split()
        logging.debug('   Validation subjs: {}'.format(training_subjs))

    # Verify that subjects exist and that no subjects are forgotten
    chosen_subjs = training_subjs + validation_subjs
    verify_subject_lists(dwi_ml_ready_folder, chosen_subjs)

    # Read group information from the json file
    json_file = open(args.config_file, 'r')
    groups_config = json.load(json_file)

    # Check if hdf5 already exists and initialize database
    hdf5_dir, hdf5_filename = _initialize_hdf5_database(args.database_folder,
                                                        args.force, args.name)

    # Check that all groups contain both "type" and "files" sub-keys
    _check_groups_config(groups_config)

    # Check that all files exist
    _check_files_presence(args, chosen_subjs, groups_config,
                          dwi_ml_ready_folder)

    # Create dataset from config and save
    _add_all_subjs_to_database(args, chosen_subjs, groups_config, hdf5_dir,
                               hdf5_filename, dwi_ml_ready_folder,
                               training_subjs, validation_subjs)


def _check_groups_config(groups_config):
    for group in groups_config.keys():
        logging.debug("Group's keys are {}"
                      .format(groups_config[group].keys()))
        if 'type' not in groups_config[group]:
            raise KeyError("Group {}'s type was not defined. It should be "
                           "the group type. So far, the only type implemented "
                           "is 'volume'. See the doc for a groups_config.json "
                           "example.".format(group))
        if 'files' not in groups_config[group]:
            raise KeyError("Group {}'s files were not defined. It should list "
                           "the files to load and concatenate for this group. "
                           "See the doc for a groups_config.json example."
                           .format(group))


def _initialize_hdf5_database(database_path, force, name):
    # Create hdf5 dir or clean existing one
    hdf5_dir = Path(database_path, "hdf5")
    if hdf5_dir.is_dir():
        if force:
            logging.info("    Careful! Deleting existing hdf5 data folder: "
                         "{}".format(hdf5_dir))
            shutil.rmtree(str(hdf5_dir))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. Use "
                                  "force to allow overwrite.".format(hdf5_dir))
    logging.debug("   Creating hdf5 directory")
    hdf5_dir.mkdir()

    # Define dabase name
    if name:
        hdf5_filename = name
    else:
        hdf5_filename = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))

    return hdf5_dir, hdf5_filename


def _check_files_presence(args, chosen_subjs: List[str], groups_config: Dict,
                          dwi_ml_dir):
    """The list of files to verify is :
     - the standardization mask
     - all files in the group_config file
     - the bundles
    """
    logging.debug("   Verifying files presence")

    # concatenating file_lists from all groups files:
    # See the doc for more explanation.
    config_file_list = sum(nested_lookup('files', groups_config), [])

    for subj_id in chosen_subjs:
        subj_input_dir = dwi_ml_dir.joinpath(subj_id)

        # Find subject's standardization mask
        if args.std_mask:
            subj_std_mask_file = subj_input_dir.joinpath(args.std_mask)
            if not subj_std_mask_file.is_file():
                raise FileNotFoundError("Standardization mask {} not found "
                                        "for subject {}!"
                                        .format(subj_std_mask_file, subj_id))

        # Find subject's files from group_config
        for f in config_file_list:
            this_file = subj_input_dir.joinpath(f)
            if not this_file.is_file():
                raise FileNotFoundError("File from groups_config ({}) not "
                                        "found for subject {}!"
                                        .format(f, subj_id))

        # Find streamlines
        bundles_dir = subj_input_dir.joinpath("bundles")
        if args.bundles is None:
            # Take everything found in subject bundle folder
            bundles = [str(p) for p in bundles_dir.glob('*')]
            if len(bundles) == 0:
                raise ValueError("No bundle found in the bundles folder for "
                                 "subject {}!".format(subj_id))
        else:
            for bundle_name in args.bundles:
                # Completing name
                # (for instance if no extension was given)
                # (or to allow suffixes, ex: OL could become OL_m)
                bundle_name = str(bundles_dir.joinpath(bundle_name + '*'))
                bundle_complete_name = glob.glob(bundle_name)
                if len(bundle_complete_name) == 0 & \
                        args.enforce_bundles_presence:
                    raise FileNotFoundError("Bundle {} was not found for "
                                            "subject {}!"
                                            .format(bundle_name, subj_id))
                elif len(bundle_complete_name) > 1:
                    raise ValueError("More than one file with name {} for "
                                     "subject {}. Clean your bundles folder "
                                     "or be more specific in your bundles "
                                     "list.".format(bundle_name, subj_id))


def _add_all_subjs_to_database(args, chosen_subjs: List[str],
                               groups_config: Dict, hdf5_dir, hdf5_filename,
                               dwi_ml_dir, training_subjs, validation_subjs):
    """
    Generate a dataset from a group of dMRI subjects with multiple bundles.
    All data from each group are concatenated.
    All bundles are merged as a single whole-brain dataset in voxel space.
    If wished, all intermediate steps are saved on disk in the hdf5 folder.
    """
    # Cast space from str to Space
    space = Space(args.space)

    # Add data to database
    hdf5_file_path = hdf5_dir.joinpath("{}.hdf5".format(hdf5_filename))
    with h5py.File(hdf5_file_path, 'w') as hdf_file:
        # Save version and configuration
        hdf_file.attrs['version'] = 1
        now = datetime.datetime.now()
        hdf_file.attrs['data_and_time'] = now.strftime('%d %B %Y %X')
        hdf_file.attrs['chosen_subjs'] = chosen_subjs
        hdf_file.attrs['groups_config'] = str(groups_config)
        hdf_file.attrs['training_subjs'] = training_subjs
        hdf_file.attrs['validation_subjs'] = validation_subjs

        if args.step_size is not None:
            hdf_file.attrs['step_size'] = args.step_size
        else:
            hdf_file.attrs['step_size'] = 'Not defined by user'
        if args.bundles is not None:
            hdf_file.attrs['bundles'] = [b for b in args.bundles]
        else:
            hdf_file.attrs['bundles'] = "All bundles in subjects's folders"
        hdf_file.attrs['space'] = args.space

        # Add data one subject at the time
        nb_subjs = len(chosen_subjs)
        logging.debug("   Processing {} subjects : {}"
                      .format(nb_subjs, chosen_subjs))
        nb_processed = 0
        for subj_id in chosen_subjs:
            nb_processed = nb_processed + 1
            logging.info("       *Processing subject {}/{}: {}"
                         .format(nb_processed, nb_subjs, subj_id))
            subj_input_dir = dwi_ml_dir.joinpath(subj_id)

            # Add subject to hdf database
            subj_hdf = hdf_file.create_group(subj_id)

            # Prepare subject folder for intermediate files
            subj_intermediate_path = hdf5_dir.joinpath(subj_id)
            if args.save_intermediate:
                subj_intermediate_path.mkdir()

            # Find subject's standardization mask
            if args.std_mask:
                logging.info("       - Loading mask")
                subj_std_mask_file = subj_input_dir.joinpath(args.std_mask)
                subj_std_mask_img = nib.load(subj_std_mask_file)
                subj_std_mask_data = np.asanyarray(subj_std_mask_img.dataobj)
                subj_std_mask_data = subj_std_mask_data.astype(np.bool)
            else:
                subj_std_mask_data = None

            # Add the subj data based on groups in the json config file
            # (inputs and others. All nifti)
            for group in groups_config:
                logging.info("       - Processing group '{}'...".format(group))
                file_list = groups_config[group]['files']
                group_data, group_affine, group_header = process_group(
                    group, file_list, args.save_intermediate, subj_input_dir,
                    subj_intermediate_path, subj_std_mask_data)
                logging.debug('      *Done. Now creating dataset from group.')
                hdf_group = subj_hdf.create_group(group)
                hdf_group.create_dataset('data', data=group_data)
                logging.debug('      *Done.')

                # Saving data information.
                subj_hdf[group].attrs['affine'] = group_affine
                subj_hdf[group].attrs['type'] = groups_config[group]['type']

            # Add the streamlines data
            # Header is the last group header if .tck, or 'same' if .trk
            logging.info('       - Processing bundles...')
            sft, lengths = process_streamlines(
                subj_input_dir.joinpath("bundles"), args.bundles, group_header,
                args.step_size, space)
            streamlines = sft.streamlines

            # Save streamlines
            if args.save_intermediate:
                logging.info('       - Saving intermediate tractogram.')
                save_tractogram(sft,
                                str(subj_intermediate_path.joinpath(
                                    "{}_all_streamlines.tck".format(subj_id))))

            if streamlines is None:
                logging.warning('Careful! Total tractogram for subject {} '
                                'contained no streamlines!'.format(subj_id))
            else:
                streamlines_group = subj_hdf.create_group('streamlines')
                streamlines_group.attrs['type'] = 'streamlines'

                # The hdf5 can only store numpy arrays (it is actually the
                # reason why it can fetch only precise streamlines from their
                # ID). We need to deconstruct the sft and store all its data
                # separately to allow reconstructing it later.
                (a, d, vs, vo) = sft.space_attributes
                streamlines_group.attrs['space'] = str(sft.space)
                streamlines_group.attrs['affine'] = a
                streamlines_group.attrs['dimensions'] = d
                streamlines_group.attrs['voxel_sizes'] = vs
                streamlines_group.attrs['voxel_order'] = vo

                if len(sft.data_per_point) > 0:
                    logging.warning('sft contained data_per_point. Data not '
                                    'kept.')
                if len(sft.data_per_streamline) > 0:
                    logging.warning('sft contained data_per_streamlines. '
                                    'Data not kept.')

                # Accessing private Dipy values, but necessary.
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
