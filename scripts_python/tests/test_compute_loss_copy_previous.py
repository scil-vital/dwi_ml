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
    experiments_path = tmp_path_factory.mktemp("experiments_l2t")
    return str(experiments_path)


def test_running(script_runner, experiments_path):
    hdf5_file = os.path.join(data_dir, 'hdf5_file.hdf5')
    subj_id = TEST_EXPECTED_SUBJ_NAMES[0]
    streamline_group_name = TEST_EXPECTED_STREAMLINE_GROUPS[0]

    out_tractogram = os.path.join(experiments_path, 'colored_tractogram.trk')
    out_displacement = os.path.join(experiments_path, 'displacement.trk')
    ret = script_runner.run('dwiml_compute_loss_copy_previous.py',
                            '--dg_key', 'sphere-classification',
                            '--step_size', '0.5', '--subset', 'training',
                            '--save_colored_tractogram', out_tractogram,
                            '--save_displacement', out_displacement,
                            '--pick_at_random',
                            hdf5_file, subj_id, streamline_group_name)
    assert ret.success

