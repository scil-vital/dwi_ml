#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import pytest
import tempfile

import torch

from dwi_ml.tests.utils.expected_values import (
    TEST_EXPECTED_VOLUME_GROUPS, TEST_EXPECTED_STREAMLINE_GROUPS,
    TEST_EXPECTED_SUBJ_NAMES)
from dwi_ml.tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()
MAX_LEN = 400  # During tracking, if we allow 200mm * 0.5 step size = 400 points.


def test_help_option(script_runner):
    ret = script_runner.run('tto_train_model.py', '--help')
    assert ret.success

    ret = script_runner.run(
        'tto_resume_training_from_checkpoint.py', '--help')
    assert ret.success

    ret = script_runner.run('tto_track_from_model.py', '--help')
    assert ret.success


@pytest.fixture(scope="session")
def experiments_path(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments_tto")
    return str(experiments_path)


def test_execution(script_runner, experiments_path):
    experiment_name = 'test_experiment'
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    # Here, testing default values only. See dwi_ml.tests.test_trainer for more
    # various testing.
    # Max length in current testing dataset is 108. Setting max length to 115
    # for faster testing. Also decreasing other default values.
    logging.info("************ TESTING TRAINING ************")
    ret = script_runner.run('tto_train_model.py',
                            experiments_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '2',
                            '--max_batches_per_epoch_validation', '1',
                            '--nheads', '2', '--max_len', str(MAX_LEN),
                            '--d_model', '6', '--n_layers_e', '1',
                            '--n_layers_d', '1', '--ffnn_hidden_size', '3',
                            '--dropout_rate', '0', '--logging', 'INFO',
                            '--SOS_as_label')
    assert ret.success

    logging.info("************ TESTING RESUMING FROM CHECKPOINT ************")
    ret = script_runner.run(
        'tto_resume_training_from_checkpoint.py',
        experiments_path, 'test_experiment', '--new_max_epochs', '2')
    assert ret.success

    # Test training GPU
    if torch.cuda.is_available():
        logging.info("************ TESTING TRAINING GPU ************")
        ret = script_runner.run(
            'tto_train_model.py', experiments_path, 'tto_test', hdf5_file,
            input_group_name, streamline_group_name, '--max_epochs', '1',
            '--batch_size_training', '5',
            '--batch_size_units', 'nb_streamlines',
            '--max_batches_per_epoch_training', '2',
            '--max_batches_per_epoch_validation', '1', '--nheads', '2',
            '--max_len', str(MAX_LEN),
            '--d_model', '6', '--n_layers_e', '1', '--n_layers_d', '1',
            '--ffnn_hidden_size', '3', '--dropout_rate', '0.',
            '--logging', 'INFO', '--use_gpu',
            '--SOS_as_class', 'repulsion100')
        assert ret.success

    logging.info("************ TESTING TRACKING FROM MODEL ************")
    whole_experiment_path = os.path.join(experiments_path, experiment_name)
    out_tractogram = os.path.join(tmp_dir.name, 'test_tractogram.trk')

    seeding_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    tracking_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    input_group = TEST_EXPECTED_VOLUME_GROUPS[0]
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]

    ret = script_runner.run(
        'tto_track_from_model.py', whole_experiment_path, hdf5_file, subj_id,
        out_tractogram, seeding_mask_group, tracking_mask_group, input_group,
        '--algo', 'det', '--nt', '2', '--rng_seed', '0',
        '--min_length', '0', '--subset', 'training', '--logging', 'DEBUG')

    assert ret.success
