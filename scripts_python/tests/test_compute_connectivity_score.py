#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from dwi_ml.unit_tests.utils.data_and_models_for_tests import \
    fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_connectivity_score.py', '--help')
    assert ret.success


def test_execution(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))
    dwi_ml_folder = os.path.join(data_dir, 'dwi_ml_ready', 'subjX')

    # Currently no matrix in our tests. Creating one.
    in_volume = os.path.join(dwi_ml_folder, 'anat', 't1.nii.gz')
    streamlines = os.path.join(dwi_ml_folder, 'example_bundle', 'Fornix.trk')
    matrix = 'matrix_connectivity.npy'

    nb_blocs = '4'
    script_runner.run('dwiml_compute_connectivity_matrix_from_blocs.py',
                      in_volume, streamlines, matrix, nb_blocs, '--binary')

    # Now scoring
    ret = script_runner.run('dwiml_compute_connectivity_score.py',
                            matrix, matrix)
    assert ret.success
