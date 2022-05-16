#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

#fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
from dwi_ml.tests.expected_values import TEST_EXPECTED_VOLUME_GROUPS, \
    TEST_EXPECTED_STREAMLINE_GROUPS

tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_train_model.py', '--help')
    assert ret.success

    ret = script_runner.run('dwiml_resume_training_from_checkpoint.py',
                            '--help')
    assert ret.success


def test_execution_bst(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    home = os.path.expanduser("~")
    experiment_path = tmp_dir.name
    experiment_name = 'test_experiment'
    hdf5_file = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')
    input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    # Here, testing default values only. See dwi_ml.tests.test_trainer for more
    # various testing.
    ret = script_runner.run('dwiml_train_model.py',
                            experiment_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--max_epochs', '1', '--batch_size', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--logging', 'INFO')
    assert ret.success

    # Then test running from checkpoint:
    ret = script_runner.run('dwiml_resume_training_from_checkpoint.py',
                            experiment_path, 'test_experiment',
                            '--new_max_epochs', '2')
    assert ret.success

