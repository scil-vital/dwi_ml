#!/usr/bin/env python
from datetime import datetime
import logging
import os
import tempfile

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_STREAMLINE_GROUPS, TEST_EXPECTED_VOLUME_GROUPS)
from dwi_ml.tests.utils import ModelForTest

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()

logging.basicConfig(level=logging.DEBUG)
SAVE_RESULT_SFT_NII = False
ref = None


def test_trainer():
    logging.debug("\n"
                  "Unit test: Trainer\n"
                  "------------------------")

    # os.chdir(os.path.expanduser(tmp_dir.name))
    # hdf5_filename = os.path.join(get_home(), 'dwiml', 'hdf5_file.hdf5')
    home = os.path.expanduser("~")
    hdf5_filename = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')

    dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=False,
                                  lazy=False)
    dataset.load_data()

    batch_sampler, batch_loader = _create_sampler_and_loader(dataset)
    model = ModelForTest()
    trainer = _create_trainer()


def _create_sampler_and_loader(dataset):
    batch_size = 250
    batch_size_units = 'nb_streamlines'
    chunk_size = None
    step_size = None
    compress = False
    noise_size = 0.
    noise_variability = 0.
    split_ratio = 0.
    reverse_ratio = 0.
    wait_for_gpu = True

    # Initialize batch sampler
    logging.debug('\nInitializing sampler...')
    training_set = dataset.training_set


    return batch_sampler, batch_loader


def _create_trainer(batch_sampler, batch_loader, model, tmp_dir):
    trainer = DWIMLTrainerOneInput(
        batch_sampler_training=batch_sampler,
        batch_sampler_validation=None,
        batch_loader_training=batch_loader,
        batch_loader_validation=None,
        model=model, experiment_path=tmp_dir, experiment_name='test',
        comet_project=None, comet_workspace=None, from_checkpoint=False,
        learning_rate=0.001, max_batches_per_epoch=4, max_epochs=2,
        nb_cpu_processes=None, patience=None, taskman_managed=True,
        use_gpu=False, weight_decay=None)
