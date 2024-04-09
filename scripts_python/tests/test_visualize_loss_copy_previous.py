#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import pytest

from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data
from dwi_ml.unit_tests.utils.expected_values import TEST_EXPECTED_SUBJ_NAMES, \
    TEST_EXPECTED_STREAMLINE_GROUPS

data_dir = fetch_testing_data()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_loss_copy_previous.py', '--help')
    assert ret.success


@pytest.fixture(scope="session")
def experiments_path(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("experiment_copy_prev")
    return str(experiments_path)


def test_running(script_runner, experiments_path):
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    prefix = 'fornix_'
    out_dir = os.path.join(experiments_path, 'test_visu')
    ret = script_runner.run('dwiml_compute_loss_copy_previous.py',
                            hdf5_file, subj_id,
                            '--streamlines_group', streamline_group_name,
                            '--out_prefix', prefix,
                            '--out_dir', out_dir,
                            '--subset', 'training',
                            '--save_colored_tractogram',
                            '--save_colored_best_and_worst',
                            '--save_displacement', '--batch_size', '100',
                            '--min_range', '-1', '--max_range', '1',
                            '--displacement_on_nb', '1',
                            '--displacement_on_best_and_worst')
    assert ret.success
