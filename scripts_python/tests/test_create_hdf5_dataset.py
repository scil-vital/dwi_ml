#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

#fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_create_hdf5_dataset.py', '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    home = os.path.expanduser("~")
    dwi_ml_folder = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/dwi_ml_ready')
    config_file = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/code_creation/config_file.json')
    training_subjs = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/code_creation/training_subjs.txt')
    validation_subjs = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/code_creation/empty_subjs_list.txt')
    testing_subjs = validation_subjs

    hdf5_output = 'test.hdf5'

    ret = script_runner.run('dwiml_create_hdf5_dataset.py',
                            dwi_ml_folder, hdf5_output, config_file,
                            training_subjs, validation_subjs, testing_subjs)
    assert ret.success
