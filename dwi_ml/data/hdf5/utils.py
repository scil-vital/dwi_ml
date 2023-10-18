# -*- coding: utf-8 -*-
import datetime
import shutil
from argparse import ArgumentParser
import json
import logging
import os
from pathlib import Path
from typing import List

from dwi_ml.data.hdf5.hdf5_creation import HDF5Creator
from dwi_ml.io_utils import add_resample_or_compress_arg


def add_nb_blocs_connectivity_arg(p: ArgumentParser):
    p.add_argument(
        '--connectivity_nb_blocs', metavar='m', type=int, nargs='+',
        help="Number of 3D blocks (m x m x m) for the connectivity matrix. \n"
             "(The matrix will be m^3 x m^3). If more than one values are "
             "provided, expected to be one per dimension. \n"
             "Default: 20x20x20.")


def format_nb_blocs_connectivity(connectivity_nb_blocs) -> List:
    if connectivity_nb_blocs is None:
        # Default/const value with argparser '+' not possible.
        # Setting it manually.
        connectivity_nb_blocs = 20
    elif len(connectivity_nb_blocs) == 1:
        connectivity_nb_blocs = connectivity_nb_blocs[0]

    if isinstance(connectivity_nb_blocs, List):
        assert len(connectivity_nb_blocs) == 3, \
            "Expecting to work with 3D volumes. Expecting " \
            "connectivity_nb_blocs to be a list of 3 values, " \
            "but got {}.".format(connectivity_nb_blocs)
    else:
        assert isinstance(connectivity_nb_blocs, int), \
            "Expecting the connectivity_nb_blocs to be either " \
            "a 3D list or an integer, but got {}" \
            .format(connectivity_nb_blocs)
        connectivity_nb_blocs = [connectivity_nb_blocs] * 3

    return connectivity_nb_blocs


def add_hdf5_creation_args(p: ArgumentParser):

    # Positional arguments
    p.add_argument('dwi_ml_ready_folder',
                   help="Path to the folder containing the data. \n"
                        "-> Should follow description in our doc, here: \n"
                        "-> https://dwi-ml.readthedocs.io/en/latest/"
                        "creating_hdf5.html")
    p.add_argument('out_hdf5_file',
                   help="Path and name of the output hdf5 file.\n If "
                        "--save_intermediate is set, the intermediate files "
                        "will be saved in \nthe same location, in a folder "
                        "name based on date and hour of creation.\n"
                        "If it already exists, use -f to allow overwriting.")
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

    # Optional arguments
    p.add_argument('--enforce_files_presence', type=bool, default=True,
                   metavar="True/False",
                   help='If True, the process will stop if one file is '
                        'missing for a subject. \nChecks are not made for '
                        'option "ALL" for streamline groups.\nDefault: True')
    p.add_argument('--save_intermediate', action="store_true",
                   help="If set, save intermediate processing files for "
                        "each subject inside the \nhdf5 folder, in sub-"
                        "folders named subjid_intermediate.")

    p.add_argument('--logging',
                   choices=['error', 'warning', 'info', 'debug'],
                   default='warning',
                   help="Logging level. [warning]")


def add_mri_processing_args(p: ArgumentParser):
    g = p.add_argument_group('Volumes processing options:')
    g.add_argument(
        '--std_mask', nargs='+', metavar='m',
        help="Mask defining the voxels used for data standardization. \n"
             "-> Should be the name of a file inside dwi_ml_ready/{subj_id}.\n"
             "-> You may add wildcards (*) that will be replaced by the "
             "subject's id. \n"
             "-> If none is given, all non-zero voxels will be used.\n"
             "-> If more than one are given, masks will be combined.")


def add_streamline_processing_args(p: ArgumentParser):
    g = p.add_argument_group('Streamlines processing options:')
    add_resample_or_compress_arg(g)

    gc = g.add_mutually_exclusive_group()
    gc.add_argument(
        '--compute_connectivity_from_blocs', action='store_true',
        help="If set, computes the 3D connectivity matrix for each streamline "
             "group. \nDefined from image split into blocs, not from anatomy!\n"
             "Ex: can be used at validation time with our trainer's "
             "'generation-validation' step. \nSee connectivity_nb_blocs.")
    gc.add_argument(
        '--compute_connectivity_from_labels', metavar='label_group',
        help="If set, computes de 3D connectivity matrix for each streamline "
             "group using labels from one volume group (in the config file).")
    add_nb_blocs_connectivity_arg(g)


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
                          args.compress, args.compute_connectivity_from_blocs,
                          args.compute_connectivity_from_labels,
                          args.connectivity_nb_blocs,
                          args.enforce_files_presence,
                          args.save_intermediate, intermediate_subdir)

    return creator
