# -*- coding: utf-8 -*-
import datetime
import json
import logging
import shutil
from pathlib import Path

from dipy.io.stateful_tractogram import Space

from dwi_ml.data.hdf5.hdf5_creation import HDF5Creator


def add_basic_args(p):
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
    p.add_argument('--name',
                   help="Dataset name [Default uses date and time of "
                        "processing].")
    p.add_argument('--enforce_files_presence', type=bool, default=True,
                   metavar="True/False",
                   help='If True, the process will stop if one file is '
                        'missing for a subject. \nDefault: True')
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


def add_mri_processing_args(p):
    p.add_argument(
        '--std_mask', nargs='*',
        help="Mask defining the voxels used for data standardization. \n"
             "-> Should be the name of a file inside dwi_ml_ready/{subj_id}.\n"
             "-> You may add wildcards (*) that will be replaced by the "
             "subject's id. \n"
             "-> If none is given, all non-zero voxels will be used.\n"
             "-> If more than one are given, masks will be combined.")


def add_streamline_processing_args(p):
    g = p.add_mutually_exclusive_group()
    g.add_argument('--step_size', type=float, metavar='s',
                   help="Common step size to resample the data. \n"
                        "-> Must be in the same space as the streamlines.")
    g.add_argument('--compress', action="store_true",
                   help="If set, streamlines will be compressed.\n"
                        "-> If neither step_size nor compress are chosen, "
                        "streamlines will be kept \nas they are.")
    p.add_argument('--space', type=str, default='vox',
                   choices=['rasmm', 'vox', 'voxmm'],
                   help="Default space to bring all the stateful tractograms.")


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


def prepare_hdf5_creator(args):
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

    return creator
