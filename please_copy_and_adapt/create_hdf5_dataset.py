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
import logging
from pathlib import Path

from dipy.io.stateful_tractogram import set_sft_logger_level

from dwi_ml.data.hdf5.utils import (
    add_basic_args, add_mri_processing_args, add_streamline_processing_args,
    prepare_hdf5_creator)
from dwi_ml.experiment_utils.timer import Timer


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    add_basic_args(p)
    add_mri_processing_args(p)
    add_streamline_processing_args(p)

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

    creator = prepare_hdf5_creator(args)

    # Create dataset from config and save
    with Timer("\nCreating database...", newline=True, color='green'):
        creator.create_database()


if __name__ == '__main__':
    main()
