#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

from dwi_ml.tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_compute_loss_copy_previous.py', '--help')
    assert ret.success


def test_running(script_runner):
    sft = os.path.join(data_dir, 'dwi_ml_ready/subjX/example_bundle/Fornix.trk')
    ret = script_runner.run('dwiml_compute_loss_copy_previous.py', sft,
                            '--dg_key', 'sphere-classification',
                            '--step_size', '0.5')
    assert ret.success

