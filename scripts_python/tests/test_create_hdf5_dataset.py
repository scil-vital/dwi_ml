#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from dwi_ml.tests.utils import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_create_hdf5_dataset.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    dwi_ml_folder = os.path.join(data_dir, 'dwi_ml_ready')
    config_file = os.path.join(data_dir, 'code_creation/config_file.json')
    training_subjs = os.path.join(data_dir, 'code_creation/training_subjs.txt')
    validation_subjs = os.path.join(data_dir,
                                    'code_creation/empty_subjs_list.txt')
    testing_subjs = validation_subjs

    hdf5_output = 'test.hdf5'

    ret = script_runner.run('dwiml_create_hdf5_dataset.py',
                            dwi_ml_folder, hdf5_output, config_file,
                            training_subjs, validation_subjs, testing_subjs)
    assert ret.success
