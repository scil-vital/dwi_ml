#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
import tempfile

from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()


def _download_ismrm():
    url = "https://scil.usherbrooke.ca/ismrm2015/scoring_data_Renauld2023.zip"
    result = subprocess.run(["wget", url], capture_output=True)
    assert result.returncode == 0, \
        "Could not download ismrm data.\n {}".format(result.stderr)

    result = subprocess.run(["unzip", "scoring_data_Renauld2023.zip",
                             "-d", "scoring_data_Renauld2023"],
                            capture_output=True)
    assert result.returncode == 0, \
        "Could not unzip data!\n {}".format(result.stderr)


def test_score_ismrm():
    os.chdir(os.path.expanduser(tmp_dir.name))

    # Adding here a test on scil_score_ismrm.
    # Result will be ugly but at least we can check if runs properly
    _download_ismrm()

    # Real test!
    scoring_data = "./scoring_data_Renauld2023"
    tractogram = os.path.join(data_dir, 'dwi_ml_ready', 'subjX',
                              'example_bundle' 'Fornix.trk')
    out_dir = './test_ismrm'
    result = subprocess.run(['scil_score_ismrm_Renauld2023.sh',
                            tractogram, out_dir, scoring_data],
                            capture_output=True)
    assert result.returncode == 0, \
        "Failed to run scil_score_ismrm_Renauld2023.sh.\n{}".format(result.stderr)
