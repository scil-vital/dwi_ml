#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import pytest

import torch

from dwi_ml.unit_tests.utils.expected_values import (
    TEST_EXPECTED_VOLUME_GROUPS,
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_SUBJ_NAMES)
from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
experiment_name = 'test_experiment'
hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]


def test_help_option(script_runner):
    ret = script_runner.run('l2t_train_model', '--help')
    assert ret.success

    ret = script_runner.run('l2t_resume_training_from_checkpoint', '--help')
    assert ret.success

    ret = script_runner.run('l2t_track_from_model', '--help')
    assert ret.success

    ret = script_runner.run('l2t_visualize_loss', '--help')
    assert ret.success

    ret = script_runner.run('l2t_update_deprecated_exp', '--help')
    assert ret.success


@pytest.fixture(scope="session")
def experiments_path(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiments_l2t")
    return str(experiments_path)


def test_training(script_runner, experiments_path):
    # Here, testing default values only. See dwi_ml.unit_tests.test_trainer for
    # more various testing.
    logging.info("************ TESTING TRAINING ************")
    ret = script_runner.run('l2t_train_model',
                            experiments_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--add_relative_coords_to_input',
                            '--input_embedding_key', 'nn_embedding',
                            '--input_embedded_size', '8',
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_validation', '5',
                            '--dg_key', 'gaussian',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '2',
                            '--max_batches_per_epoch_validation', '1',
                            '-v', 'INFO', '--step_size', '0.5', '--add_eos',
                            '--nb_previous_dirs', '1')
    assert ret.success

    # Test training on GPU
    if torch.cuda.is_available():
        logging.info("************ TESTING TRAINING GPU ************")
        ret = script_runner.run('l2t_train_model',
                                experiments_path, 'test_l2t_gpu', hdf5_file,
                                input_group_name, streamline_group_name,
                                '--max_epochs', '1', '--step_size', '0.5',
                                '--learning_rate', '0.002',
                                '--batch_size_training', '5',
                                '--batch_size_validation', '5',
                                '--batch_size_units', 'nb_streamlines',
                                '--max_batches_per_epoch_training', '2',
                                '--max_batches_per_epoch_validation', '1',
                                '--dg_key', 'cosine-regression', '--add_eos',
                                '-v', 'INFO', '--use_gpu')
        assert ret.success


def test_checkpoint(script_runner, experiments_path):
    logging.info("************ TESTING RESUMING FROM CHECKPOINT ************")
    ret = script_runner.run('l2t_resume_training_from_checkpoint',
                            experiments_path, 'test_experiment',
                            '--new_max_epochs', '2')
    assert ret.success


def test_tracking(script_runner, experiments_path):
    # Test Tracking
    logging.info("************ TESTING TRACKING FROM MODEL ************")
    whole_experiment_path = os.path.join(experiments_path, experiment_name)
    out_tractogram = os.path.join(experiments_path, 'test_tractogram.trk')

    seeding_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    tracking_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]
    # Testing HDF5 data does not contain a testing set to keep it light. Using
    # subjectX from training set.
    ret = script_runner.run(
        'l2t_track_from_model', whole_experiment_path, subj_id,
        input_group_name, out_tractogram, seeding_mask_group,
        '--hdf5_file', hdf5_file,
        '--algo', 'det', '--nt', '2', '--rng_seed', '0', '--min_length', '0',
        '--subset', 'training', '--tracking_mask_group', tracking_mask_group)

    assert ret.success

    # Testing multiple GPU tracking
    if torch.cuda.is_available():
        logging.info("********** TESTING GPU TRACKING FROM MODEL ************")
        out_tractogram = os.path.join(experiments_path, 'test_tractogram2.trk')
        ret = script_runner.run(
            'l2t_track_from_model', whole_experiment_path,
            subj_id, input_group_name, out_tractogram, seeding_mask_group,
            '--algo', 'det', '--nt', '20', '--rng_seed', '0',
            '--min_length', '0', '--subset', 'training',
            '--tracking_mask_group', tracking_mask_group,
            # Additional params compared to CPU:
            '--use_gpu', '--simultaneous_tracking', '3')

        assert ret.success


def test_visu(script_runner, experiments_path):
    whole_experiment_path = os.path.join(experiments_path, experiment_name)
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]

    # Test visu loss
    prefix = 'fornix_'
    ret = script_runner.run('l2t_visualize_loss', whole_experiment_path,
                            hdf5_file, subj_id, input_group_name,
                            '--streamlines_group', streamline_group_name,
                            '--out_prefix', prefix,
                            '--subset', 'training',
                            '--compute_histogram',
                            '--save_colored_tractogram',
                            '--save_colored_best_and_worst', '1',
                            '--save_colored_eos_probs',
                            '--save_colored_eos_errors',
                            '--save_displacement', '1', '--batch_size', '100',
                            '--min_range', '-1', '--max_range', '1')
    assert ret.success


def test_training_with_generation_validation(script_runner, experiments_path):

    logging.info("************ TESTING TRAINING WITH GENERATION ************")
    # Note. No connectivity in the data! To add!
    experiment_name = 'test2'
    ret = script_runner.run('l2t_train_model',
                            experiments_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_validation', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '2',
                            '--max_batches_per_epoch_validation', '1',
                            '-v', 'INFO', '--step_size', '0.5',
                            '--add_a_tracking_validation_phase',
                            '--tracking_mask', 'wm_mask',
                            '--tracking_phase_frequency', '1')
    assert ret.success
