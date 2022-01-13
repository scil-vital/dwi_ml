#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to combine multiple diffusion MRI volumes and their streamlines into a
single .hdf5 file.

You should have a file dwi_ml_ready organized as prescribed (see our doc for
more information.) A hdf5 folder will be created alongside dwi_ml_ready.
It will contain the .hdf5 file and possibly intermediate files.

Important notes
---------------

The memory is a delicate question here, but checks have been made, and it
appears that the SFT's garbage collector may not be working entirely well.

Keeping as is for now, hoping that next Dipy versions will solve the problem.
"""

import argparse
import datetime
import json
import logging
import shutil
from pathlib import Path

from dipy.io.stateful_tractogram import set_sft_logger_level, Space

# Modify this class to fit your needs.
from dwi_ml.data.hdf5_creation import HDF5Creator
from dwi_ml.experiment_utils.timer import Timer


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    # Positional arguments
    p.add_argument('dwi_ml_ready_folder',
                   help="Path to the folder containing the data. \n"
                        "-> Should follow description in our doc, here: \n"
                        "-> https://dwi-ml.readthedocs.io/en/latest/"
                        "creating_hdf5.html")
    p.add_argument('hdf5_folder',
                   help="Path where to save the hdf5 database and possibly "
                        "the intermediate files. \nWe will create a sub-"
                        "folder /hdf5 inside. \nIf it already exists, use -f "
                        "to allow deleting and creating a new one.")
    p.add_argument('config_file',
                   help="Path to the json config file defining the groups "
                        "wanted in your hdf5. \n"
                        "-> Should follow description in our doc, here: \n"
                        "-> https://dwi-ml.readthedocs.io/en/latest/"
                        "creating_hdf5.html")
    p.add_argument('training_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for training.")
    p.add_argument('validation_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for validation.")
    p.add_argument('testing_subjs',
                   help="txt file containing the list of subjects ids to use "
                        "for testing.")
    g = p.add_mutually_exclusive_group()
    g.add_argument('--step_size', type=float, metavar='s',
                   help="Common step size to resample the data. \n"
                        "-> Must be in the same space as the streamlines.")
    g.add_argument('--compress', action="store_true",
                   help="If set, streamlines will be compressed.\n"
                        "-> If neither step_size nor compress are chosen, "
                        "streamlines will be kept \nas they are.")
    p.add_argument('--name',
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--enforce_files_presence', type=bool, default=True,
                   metavar="True/False",
                   help='If True, the process will stop if one file is '
                        'missing for a subject. \nDefault: True')
    p.add_argument(
        '--std_mask', nargs='*',
        help="Mask defining the voxels used for data standardization. \n"
             "-> Should be the name of a file inside dwi_ml_ready/{subj_id}.\n"
             "-> You may add wildcards (*) that will be replaced by the "
             "subject's id. \n"
             "-> If none is given, all non-zero voxels will be used.\n"
             "-> If more than one are given, masks will be combined.")
    p.add_argument('--space', type=str, default='vox',
                   choices=['rasmm', 'vox', 'voxmm'],
                   help="Default space to bring all the stateful tractograms.")
    p.add_argument('--logging',
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Choose logging level. [warning]")
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject inside the \nhdf5 folder, in sub-"
                        "folders named subjid_intermediate.")
    p.add_argument('-f', '--force', action='store_true',
                   help="If set, overwrite existing hdf5 folder.")

    arguments = p.parse_args()

    return arguments


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    args = _parse_args()

    # Initialize logger
    logging.basicConfig(level=str(args.logging).upper())

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Verify that dwi_ml_ready folder is found
    if not Path(args.dwi_ml_ready_folder).is_dir():
        raise ValueError('The dwi_ml_ready folder was not found: {}'
                         .format(args.dwi_ml_ready_folder))
    else:
        logging.info('***Creating hdf5 dataset')
        logging.debug('   Creating hdf5 data from data in the following foler:'
                      ' {}'.format(args.dwi_ml_ready_folder))

    # Read training subjs and validation subjs
    with open(args.training_subjs, 'r') as file:
        training_subjs = file.read().split()
        logging.debug('   Training subjs: {}'.format(training_subjs))
    with open(args.validation_subjs, 'r') as file:
        validation_subjs = file.read().split()
        logging.debug('   Validation subjs: {}'.format(validation_subjs))
    with open(args.testing_subjs, 'r') as file:
        testing_subjs = file.read().split()
        logging.debug('   Testing subjs: {}'.format(testing_subjs))

    # Read group information from the json file
    json_file = open(args.config_file, 'r')
    groups_config = json.load(json_file)

    # Check if hdf5 already exists and initialize database
    hdf5_subdir, hdf5_filename = _initialize_hdf5_database(
        args.hdf5_folder, args.force, args.name)

    hdf5_file_path = hdf5_subdir.joinpath("{}.hdf5".format(hdf5_filename))

    # Instantiate a creator and perform checks
    creator = HDF5Creator(Path(args.dwi_ml_ready_folder), hdf5_file_path,
                          training_subjs, validation_subjs, testing_subjs,
                          groups_config, args.std_mask, args.step_size,
                          args.compress, Space(args.space),
                          args.enforce_files_presence,
                          args.save_intermediate)

    # Create dataset from config and save
    with Timer("\nCreating database...", newline=True, color='green'):
        creator.create_database()


def _initialize_hdf5_database(hdf5_folder, force, name):
    # Create hdf5 dir or clean existing one
    hdf5_subdir = Path(hdf5_folder, "hdf5")
    if hdf5_subdir.is_dir():
        if force:
            logging.info("    Careful! Deleting existing hdf5 data folder: "
                         "{}".format(hdf5_subdir))
            shutil.rmtree(str(hdf5_subdir))
        else:
            raise FileExistsError("hdf5 data folder already exists: {}. Use "
                                  "force to allow overwrite."
                                  .format(hdf5_subdir))
    logging.debug("   Creating hdf5 directory")
    hdf5_subdir.mkdir()

    # Define database name
    if name:
        hdf5_filename = name
    else:
        hdf5_filename = "{}".format(
            datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S"))

    return hdf5_subdir, hdf5_filename


if __name__ == '__main__':
    main()
