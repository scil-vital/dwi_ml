#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('scil_apply_transform_to_tractogram.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    std_mask = "masks/wm.nii.gz"
    dwi_ml_folder = os.path.join(get_home(), 'dwiml', 'dwi_ml_ready')
    hdf5_output = 'test.hdf5'

    current_path = os.path.dirname(os.path.abspath(__file__))
    config_file = os.path.join(current_path, 'config_file_for_tests.json')
    training_subjs = os.path.join(current_path, 'training_subjs.txt')
    validation_subjs = os.path.join(current_path, 'validation_subjs.txt')
    testing_subjs = os.path.join(current_path, 'testing_subjs.txt')

    ret = script_runner.run('dwiml_create_hdf5_dataset.py',
                            '--std_mask', std_mask,
                            dwi_ml_folder, hdf5_output, config_file,
                            training_subjs, validation_subjs, testing_subjs)
    assert ret.success
