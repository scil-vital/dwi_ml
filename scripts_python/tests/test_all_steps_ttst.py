#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import pytest
import tempfile

import torch

from dwi_ml.unit_tests.utils.expected_values import \
    (TEST_EXPECTED_VOLUME_GROUPS, TEST_EXPECTED_STREAMLINE_GROUPS,
     TEST_EXPECTED_SUBJ_NAMES)
from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()
MAX_LEN = 400  # During tracking, if we allow 200mm * 0.5 step size = 400 points.


def test_help_option(script_runner):
    ret = script_runner.run('ttst_visualize_loss.py', '--help')
    assert ret.success


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
                            '--model', 'TTST',
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '2',
                            '--max_batches_per_epoch_validation', '1',
                            '--nheads', '2', '--max_len', str(MAX_LEN),
                            '--input_embedding_key', 'nn_embedding',
                            '--input_embedded_size', '6',
                            '--target_embedded_size', '2',
                            '--n_layers_e', '1',
                            '--ffnn_hidden_size', '3', '--logging', 'INFO')
    assert ret.success

    if torch.cuda.is_available():
        logging.info("************ TESTING TRAINING GPU ************")
        ret = script_runner.run('tt_train_model.py',
                                experiments_path, 'ttst_test', hdf5_file,
                                input_group_name, streamline_group_name,
                                '--model', 'TTST',
                                '--max_epochs', '1', '--batch_size_training', '5',
                                '--batch_size_units', 'nb_streamlines',
                                '--max_batches_per_epoch_training', '2',
                                '--max_batches_per_epoch_validation', '1',
                                '--nheads', '2', '--max_len', str(MAX_LEN),
                                '--input_embedding_key', 'nn_embedding',
                                '--input_embedded_size', '6',
                                '--target_embedded_size', '2',
                                '--n_layers_e', '1',
                                '--ffnn_hidden_size', '3', '--logging', 'INFO',
                                '--use_gpu')
        assert ret.success

    logging.info("************ TESTING TRACKING FROM MODEL ************")
    whole_experiment_path = os.path.join(experiments_path, experiment_name)

    input_group = TEST_EXPECTED_VOLUME_GROUPS[0]
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]

    # Test visu loss
    out_tractogram = os.path.join(experiments_path, 'colored_tractogram.trk')
    out_displacement = os.path.join(experiments_path, 'displacement.trk')
    ret = script_runner.run('ttst_visualize_loss.py', whole_experiment_path,
                            hdf5_file, subj_id, '--subset', 'training',
                            '--save_colored_tractogram', out_tractogram,
                            '--save_displacement', out_displacement,
                            '--min_range', '-1', '--max_range', '1',
                            '--pick_at_random')
    assert ret.success

    logging.info("************ TESTING VISUALIZE WEIGHTS ************")
    in_sft = os.path.join(data_dir, 'dwi_ml_ready/subjX/example_bundle/Fornix.trk')
    ret = script_runner.run(
        'ttst_visualize_weights.py', whole_experiment_path, hdf5_file, subj_id,
        input_group, in_sft, '--step_size', '0.5',
        '--subset', 'training', '--logging', 'INFO', '--run_locally')
    assert ret.success
