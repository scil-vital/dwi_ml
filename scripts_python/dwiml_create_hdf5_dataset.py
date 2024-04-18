#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script combines multiple diffusion MRI volumes and their streamlines into
a single .hdf5 file.

--------------------------------------
See here for the complete explanation!
    - How to organize your data
    - How to prepare the config file
    - How to run this script.
    https://dwi-ml.readthedocs.io/en/latest/2_A_creating_the_hdf5.html
--------------------------------------

** Note: The memory is a delicate question here, but checks have been made, and
it appears that the SFT's garbage collector may not be working entirely well.
Keeping as is for now, hoping that next Dipy versions will solve the problem.
"""

import argparse
import datetime
import json
import logging
import os
import shutil
from pathlib import Path

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

from dipy.io.stateful_tractogram import set_sft_logger_level

from dwi_ml.data.hdf5.hdf5_creation import HDF5Creator
from dwi_ml.data.hdf5.utils import (
    add_hdf5_creation_args, add_streamline_processing_args)
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_verbose_arg


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
                          groups_config, args.step_size, args.compress,
                          args.enforce_files_presence,
                          args.save_intermediate, intermediate_subdir)

    return creator


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    add_hdf5_creation_args(p)
    add_streamline_processing_args(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    p = _parse_args()
    args = p.parse_args()

    # Initialize logger
    logging.getLogger().setLevel(level=args.verbose)

    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')

    # Verify that dwi_ml_ready folder is found
    if not Path(args.dwi_ml_ready_folder).is_dir():
        raise ValueError('The dwi_ml_ready folder was not found: {}'
                         .format(args.dwi_ml_ready_folder))
    assert_inputs_exist(p, [args.config_file],
                        [args.training_subjs, args.validation_subjs,
                         args.testing_subjs])
    # check hdf extension
    _, ext = os.path.splitext(args.out_hdf5_file)
    if ext == '':
        args.out_hdf5_file += '.hdf5'
    elif ext != '.hdf5':
        raise p.error("The hdf5 file's extension should be .hdf5, but "
                      "received {}".format(ext))
    assert_outputs_exist(p, args, args.out_hdf5_file)

    # Prepare creator and load config file.
    creator = prepare_hdf5_creator(args)

    # Create dataset from config and save
    with Timer("\nCreating database...", newline=True, color='green'):
        creator.create_database()


if __name__ == '__main__':
    main()
