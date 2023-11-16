#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import pytest
import tempfile

from dwi_ml.unit_tests.utils.expected_values import \
    (TEST_EXPECTED_VOLUME_GROUPS, TEST_EXPECTED_STREAMLINE_GROUPS)
from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()
MAX_LEN = 400  # During tracking, if we allow 200mm * 0.5 step size = 400 points.


def test_help_option(script_runner):
    pass


@pytest.fixture(scope="session")
def experiments_path(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments_ttst")
    return str(experiments_path)


def test_execution(script_runner, experiments_path):
    experiment_name = 'test_experiment'
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    # Here, testing default values only. See dwi_ml.unit_tests.test_trainer for more
    # various testing.
    # Max length in current testing dataset is 108. Setting max length to 115
    # for faster testing. Also decreasing other default values.
    logging.info("************ TESTING TRAINING ************")
    ret = script_runner.run('tt_train_model.py',
                            experiments_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--model', 'TTS',
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '2',
                            '--max_batches_per_epoch_validation', '1',
                            '--nheads', '2', '--max_len', str(MAX_LEN),
                            '--input_embedding_key', 'nn_embedding',
                            '--input_embedded_size', '6',
                            '--n_layers_e', '1',
                            '--ffnn_hidden_size', '3', '--logging', 'INFO')
    assert ret.success
