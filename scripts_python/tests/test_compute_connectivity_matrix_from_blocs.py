#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_connectivity_matrix_from_blocs.py',
                            '--help')
    assert ret.success


def test_run(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    dwi_ml_folder = os.path.join(data_dir, 'dwi_ml_ready', 'subjX')
    in_volume = os.path.join(dwi_ml_folder, 'anat', 't1.nii.gz')
    streamlines = os.path.join(dwi_ml_folder, 'example_bundle', 'Fornix.trk')

    out_file = 'test_matrix_connectivity.npy'
    biggest = 'test_biggest.trk'

    nb_blocs = '4'
    ret = script_runner.run('dwiml_compute_connectivity_matrix_from_blocs.py',
                            in_volume, streamlines, out_file, nb_blocs,
                            '--binary', '--save_biggest', biggest)
    assert ret.success
