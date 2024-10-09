#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import tempfile

from dwi_ml.unit_tests.utils.data_and_models_for_tests import \
    fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def test_help_option(script_runner):
    ret = script_runner.run('dwiml_create_hdf5_dataset.py', '--help')
    assert ret.success


def _tmp_add_dps(script_runner):
    import json

    # Fake-adding dps to our tractogram.
    # ToDo Add dps in the data on the google drive. Currently, I need to do
    #  this ugly trick.
    dwi_ml_folder = os.path.join(data_dir, 'dwi_ml_ready')
    subj_folder = os.path.join(dwi_ml_folder, 'subjX', )
    in_trk = os.path.join(subj_folder, 'example_bundle', 'Fornix.trk')
    in_anat = os.path.join(subj_folder, 'anat', 't1.nii.gz')
    script_runner.run('scil_tractogram_assign_custom_color.py', in_trk,
                      in_trk, '--from_anatomy', in_anat, '-f')
    script_runner.run('scil_tractogram_dpp_math.py', 'mean', in_trk, in_trk,
                      '-f', '--mode', 'dpp', '--in_dpp_name', 'color',
                      '--out_keys', 'mean_color')
    ret = script_runner.run(
        'scil_tractogram_dpp_math.py', 'mean', in_trk, in_trk,
        '-f', '--mode', 'dps', '--in_dpp_name', 'mean_color',
        '--out_keys', 'mean_color_dps')
    assert ret.success

    # toDo. Add DPS to our config file
    config = {
        "input": {
            "type": "volume",
            "files": ["anat/t1.nii.gz", "dwi/fa.nii.gz"],
            "standardization": "per_file",
            "std_mask": ["masks/wm.nii.gz"]
        },
        "wm_mask": {
            "type": "volume",
            "files": ["masks/wm.nii.gz"],
            "standardization": "none"
        },
        "streamlines": {
            "type": "streamlines",
            "files": ["example_bundle/Fornix.trk"],
            "dps_keys": 'mean_color_dps'
        }
    }
    config_file = os.path.join(data_dir, 'code_creation/config_file.json')
    os.remove(config_file)
    with open(config_file, 'w') as json_file:
        json.dump(config, json_file)


def test_execution(script_runner):
    os.chdir(os.path.expanduser(tmp_dir.name))

    _tmp_add_dps(script_runner)
    # hdf5_output = 'test.hdf5'
    # Overwriting current hdf5!!
    hdf5_output = os.path.join(data_dir, 'hdf5_file.hdf5')

    # Note. Our test config file is:
    # {
    #     "input": {
    #          "type": "volume",
    #          "files": ["anat/t1.nii.gz", "dwi/fa.nii.gz"],
    #          "standardization": "per_file",
    #          "std_mask": ["masks/wm.nii.gz"]
    #     },
    #     "wm_mask": {
    #          "type": "volume",
    #          "files": ["masks/wm.nii.gz"],
    #          "standardization": "none"
    #     },
    #     "streamlines": {
    #          "type": "streamlines",
    #          "files": ["example_bundle/Fornix.trk"],
    #          "dps_keys": 'mean_color_dps'
    #     }
    # }
    dwi_ml_folder = os.path.join(data_dir, 'dwi_ml_ready')
    config_file = os.path.join(data_dir, 'code_creation/config_file.json')
    training_subjs = os.path.join(data_dir, 'code_creation/subjs_list.txt')
    validation_subjs = os.path.join(data_dir,
                                    'code_creation/empty_subjs_list.txt')
    testing_subjs = validation_subjs

    ret = script_runner.run('dwiml_create_hdf5_dataset.py', '-f',
                            dwi_ml_folder, hdf5_output, config_file,
                            training_subjs, validation_subjs, testing_subjs)
    assert ret.success
