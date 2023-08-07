# -*- coding: utf-8 -*-
import datetime
import shutil
from argparse import ArgumentParser
import json
import logging
import os
from pathlib import Path

from dwi_ml.data.hdf5.hdf5_creation import HDF5Creator
from dwi_ml.arg_utils import get_resample_or_compress_arg, get_overwrite_arg, \
    get_logging_arg


def get_hdf5_args_groups():
    groups = {
        'Main HDF5 properties': _main_hdf5_creation_args(),
        'Volumes processing options': _mri_processing_args(),
        'Streamlines processing options': _streamline_processing_args(),
        'Others': {}
    }
    groups['Others'].update(get_overwrite_arg())
    groups['Others'].update(get_logging_arg())

    return groups


def _main_hdf5_creation_args():
    # Positional arguments
    args = {
        'dwi_ml_ready_folder': {
            'help': "Path to the folder containing the data. Should follow \n"
                    "description in our doc, here: \n"
                    "https://dwi-ml.readthedocs.io/en/latest/creating_hdf5.html"},
        'out_hdf5_file': {
            'help': "Path and name of the output hdf5 file. \nIf "
                    "--save_intermediate is set, the intermediate files "
                    "will \nbe saved in the same location, in a folder "
                    "name based on the \ndate and hour of creation."},
        'config_file': {
            'help': "Path to the json config file defining the groups "
                    "wanted in \nyour hdf5. Should follow description in our "
                    "doc, here: \nhttps://dwi-ml.readthedocs.io/en/latest/"
                    "creating_hdf5.html"},
        'training_subjs': {
            'help': "A text file containing the list of subjects ids to use "
                    "for training."},
        'validation_subjs': {
            'help': "A text file containing the list of subjects ids to use "
                    "for validation."},
        'testing_subjs': {
            'help': "A text file containing the list of subjects ids to use "
                    "for testing."},
        '--do_not_verify_files_presence': {
            'action': 'store_false', 'dest': 'enforce_files_presence',
            'help': 'By default, the process will stop if one file is '
                    'missing for \na subject. Use this to skip. P.S. Checks are '
                    'not made for \noption "ALL" for streamline groups.'},
        '--save_intermediate': {
            'action': "store_true",
            'help': "If set, save intermediate processing files for each "
                    "subject \ninside the hdf5 folder, in sub-folders named "
                    "subjid_intermediate."
        }
    }
    return args


def _mri_processing_args():
    args = {
        '--std_mask': {
            'nargs': '+', 'metavar': 'm',
            'help': "Mask defining the voxels used for data standardization. "
                    "Should \nbe the name of a file inside each "
                    "dwi_ml_ready/{subj_id}. You \nmay add wildcards (*) that "
                    "will be replaced by the subject's id. \nIf none is "
                    "given, all non-zero voxels will be used. "
                    "If more \nthan one are given, masks will be combined."}
    }
    return args


def _streamline_processing_args():
    args = {
        '--compute_connectivity_matrix': {
            'action': 'store_true',
            'help': "If set, computes the 3D connectivity matrix for each "
                    "streamline \ngroup. Defined from downsampled image (i.e. "
                    "block, not from anatomy! \n"
                    "Hint: can be used at validation time with our trainer's \n"
                    "'generation-validation' step."},
        '--connectivity_downsample_size': {
            'metavar': 'm', 'type': int, 'nargs': '+',
            'help': "Number of 3D blocks (m x m x m) for the connectivity "
                    "matrix. \n(The matrix will be m^3 x m^3). If more than "
                    "one values are \nprovided, expected to be one per "
                    "dimension. \nDefault if not set: 20x20x20."}
    }
    args.update(get_resample_or_compress_arg())
    return args


def _initialize_intermediate_subdir(hdf5_file, save_intermediate):
    # Create hdf5 dir or clean existing one
    hdf5_folder = os.path.dirname(hdf5_file)

    # Preparing intermediate folder.
    if save_intermediate:
        now = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")
        intermediate_subdir = Path(hdf5_folder, "intermediate_" + now)
        logging.debug("   Creating intermediate files directory")
        intermediate_subdir.mkdir()

        return intermediate_subdir
    return None


def prepare_hdf5_creator(args):
    """
    Reads the config file and subjects lists files and instantiate a class of
    the HDF5Creator.
    """
    # Read subjects lists
    with open(args.training_subjs, 'r') as file:
        training_subjs = file.read().split()
        logging.debug('   Training subjs: {}'.format(training_subjs))
    with open(args.validation_subjs, 'r') as file:
        validation_subjs = file.read().split()
        logging.debug('   Validation subjs: {}'.format(validation_subjs))
    with open(args.testing_subjs, 'r') as file:
        testing_subjs = file.read().split()
        logging.debug('   Testing subjs: {}'.format(testing_subjs))

    # Read group information from the json file (config file)
    with open(args.config_file, 'r') as json_file:
        groups_config = json.load(json_file)

    # Delete existing hdf5, if -f
    if args.overwrite and os.path.exists(args.out_hdf5_file):
        os.remove(args.out_hdf5_file)

    # Initialize intermediate subdir
    intermediate_subdir = _initialize_intermediate_subdir(
        args.out_hdf5_file, args.save_intermediate)

    # Copy config file locally
    config_copy_name = os.path.splitext(args.out_hdf5_file)[0] + '.json'
    logging.info("Copying json config file to {}".format(config_copy_name))
    shutil.copyfile(args.config_file, config_copy_name)

    # Instantiate a creator and perform checks
    creator = HDF5Creator(Path(args.dwi_ml_ready_folder), args.out_hdf5_file,
                          training_subjs, validation_subjs, testing_subjs,
                          groups_config, args.std_mask, args.step_size,
                          args.compress, args.compute_connectivity_matrix,
                          args.connectivity_downsample_size,
                          args.enforce_files_presence,
                          args.save_intermediate, intermediate_subdir)

    return creator
