#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
experiment_name = 'test_experiment'


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_print_hdf5_architecture.py', '--help')
    assert ret.success


def test_execution(script_runner):
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    ret = script_runner.run('dwiml_print_hdf5_architecture.py', hdf5_file)
    assert ret.success
