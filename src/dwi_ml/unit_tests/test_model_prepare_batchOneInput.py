#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import torch

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.models.main_models import ModelWithOneInput
from dwi_ml.unit_tests.utils.data_and_models_for_tests import fetch_testing_data
from dwi_ml.unit_tests.utils.expected_values import TEST_EXPECTED_MRI_SHAPE


def test_model_batch():
    data_dir = fetch_testing_data()
    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')
    dataset = MultiSubjectDataset(hdf5_filename,
                                  lazy=False, log_level=logging.WARNING)
    dataset.load_data()

    # Will be using streamline's group 0 and input group 0.

    model = ModelWithOneInput(experiment_name='test',
                              nb_features=dataset.nb_features[0],
                              step_size=0.5, compress_lines=False)

    one_line = torch.rand(6, 3)
    batch_streamlines = [one_line, one_line]
    batch_inputs = model.prepare_batch_one_input(
        streamlines=batch_streamlines, subset=dataset.training_set,
        subj_idx=0, input_group_idx=0)
    assert len(batch_inputs) == 2
    assert np.array_equal(batch_inputs[0].shape,
                          [6, TEST_EXPECTED_MRI_SHAPE[0][-1]])


if __name__ == '__main__':
    logging.getLogger().setLevel(level='INFO')
    test_model_batch()
