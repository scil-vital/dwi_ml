#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This script combines multiple diffusion MRI volumes and their streamlines into
a single .hdf5 file. A hdf5 folder will be created alongside dwi_ml_ready. It
will contain the .hdf5 file and possibly intermediate files.

** You should have a file dwi_ml_ready organized as described in our doc:
https://dwi-ml.readthedocs.io/en/latest/data_organization.html

** You should have a config file as described in our doc:
https://dwi-ml.readthedocs.io/en/latest/config_file.html



** Note: The memory is a delicate question here, but checks have been made, and
it appears that the SFT's garbage collector may not be working entirely well.
Keeping as is for now, hoping that next Dipy versions will solve the problem.
"""

import argparse
import logging
import os
from pathlib import Path

from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist)

from dipy.io.stateful_tractogram import set_sft_logger_level

from dwi_ml.data.hdf5.utils import (
    add_hdf5_creation_args, add_mri_processing_args, add_streamline_processing_args,
    prepare_hdf5_creator)
from dwi_ml.experiment_utils.timer import Timer


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    add_hdf5_creation_args(p)
    add_mri_processing_args(p)
    add_streamline_processing_args(p)
    add_overwrite_arg(p)

    return p


def main():
    """Parse arguments, generate hdf5 dataset and save it on disk."""
    p = _parse_args()
    args = p.parse_args()

    # Initialize logger
    logging.getLogger().setLevel(level=str(args.logging).upper())

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

    # Default value with arparser '+' not possible. Setting manually.
    if args.compute_connectivity_matrix and \
            args.connectivity_downsample_size is None:
        args.connectivity_downsample_size = 20

    # Prepare creator and load config file.
    creator = prepare_hdf5_creator(args)

    # Create dataset from config and save
    with Timer("\nCreating database...", newline=True, color='green'):
        creator.create_database()


if __name__ == '__main__':
    main()
