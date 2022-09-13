#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import os
import tempfile

import torch
from dwi_ml.tests.expected_values import TEST_EXPECTED_VOLUME_GROUPS, \
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_SUBJ_NAMES
from dwi_ml.tests.utils import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('l2t_train_model.py', '--help')
    assert ret.success

    ret = script_runner.run('l2t_resume_training_from_checkpoint.py', '--help')
    assert ret.success

    ret = script_runner.run('l2t_track_from_model.py', '--help')
    assert ret.success


def test_execution_training_tracking(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    experiment_path = tmp_dir.name
    experiment_name = 'test_experiment'
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    input_group_name = TEST_EXPECTED_VOLUME_GROUPS[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    # Here, testing default values only. See dwi_ml.tests.test_trainer for more
    # various testing.
    logging.info("************ TESTING TRAINING ************")
    ret = script_runner.run('l2t_train_model.py',
                            experiment_path, experiment_name, hdf5_file,
                            input_group_name, streamline_group_name,
                            '--max_epochs', '1', '--batch_size_training', '5',
                            '--batch_size_validation', '5',
                            '--batch_size_units', 'nb_streamlines',
                            '--max_batches_per_epoch_training', '5',
                            '--max_batches_per_epoch_validation', '4',
                            '--logging', 'INFO',
                            '--nb_previous_dirs', '1')
    assert ret.success

    logging.info("************ TESTING RESUMING FROM CHECKPOINT ************")
    ret = script_runner.run('l2t_resume_training_from_checkpoint.py',
                            experiment_path, 'test_experiment',
                            '--new_max_epochs', '2')
    assert ret.success

    logging.info("************ TESTING TRACKING FROM MODEL ************")
    whole_experiment_path = os.path.join(experiment_path, experiment_name)
    out_tractogram = os.path.join(tmp_dir.name, 'test_tractogram.trk')

    seeding_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    tracking_mask_group = TEST_EXPECTED_VOLUME_GROUPS[1]
    input_group = TEST_EXPECTED_VOLUME_GROUPS[0]
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]
    # Testing HDF5 data does not contain a testing set to keep it light. Using
    # subjectX from training set.
    ret = script_runner.run(
        'l2t_track_from_model.py', whole_experiment_path, hdf5_file, subj_id,
        out_tractogram, seeding_mask_group, tracking_mask_group, input_group,
        '--algo', 'det', '--nt', '2', '--rk_order', '1', '--logging', 'DEBUG',
        '--rng_seed', '0', '--min_length', '0', '--subset', 'training')

    assert ret.success

    # Testing multiple tracking
    if torch.cuda.is_available():
        logging.info("********** TESTING GPU TRACKING FROM MODEL ************")
        out_tractogram = os.path.join(tmp_dir.name, 'test_tractogram2.trk')
        ret = script_runner.run(
            'l2t_track_from_model.py', whole_experiment_path, hdf5_file,
            subj_id, out_tractogram, seeding_mask_group, tracking_mask_group,
            input_group, '--algo', 'det', '--nt', '2', '--rk_order', '1',
            '--logging', 'DEBUG', '--rng_seed', '0', '--min_length', '0',
            '--subset', 'training',
            # Additional params compared to CPU:
            '--use_gpu', '--simultaneous_tracking', '3')

        assert ret.success
