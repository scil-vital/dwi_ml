#!/usr/bin/env python
import logging
import os
import tempfile

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.tests.utils import ModelForTest, create_test_batch_sampler, \
    create_batch_loader

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()

logging.basicConfig(level=logging.DEBUG)
SAVE_RESULT_SFT_NII = False
ref = None
batch_size = 500
batch_size_units = 'nb_streamlines'


def test_trainer():
    logging.debug("\n"
                  "Unit test: Trainer\n"
                  "------------------------")

    # os.chdir(os.path.expanduser(tmp_dir.name))
    # hdf5_filename = os.path.join(get_home(), 'dwiml', 'hdf5_file.hdf5')
    home = os.path.expanduser("~")
    hdf5_filename = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')

    # Initializing dataset
    dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=False,
                                  lazy=False)
    dataset.load_data()

    # Initializing batch sampler
    batch_sampler, batch_loader = _create_sampler_and_loader(dataset)

    # Initializing model
    model = ModelForTest()

    # Start tests
    trainer = _create_trainer(batch_sampler, batch_loader, model)
    trainer.train_and_validate()


def _create_sampler_and_loader(dataset):

    # Initialize batch sampler
    logging.debug('\nInitializing sampler...')
    batch_sampler = create_test_batch_sampler(
        dataset.training_set, batch_size=batch_size,
        batch_size_units='nb_streamlines', log_level=logging.WARNING)

    batch_loader = create_batch_loader(dataset.training_set,
                                       log_level=logging.WARNING,
                                       wait_for_gpu=False)

    return batch_sampler, batch_loader


def _create_trainer(batch_sampler, batch_loader, model):
    trainer = DWIMLTrainerOneInput(
        batch_sampler_training=batch_sampler,
        batch_sampler_validation=None,
        batch_loader_training=batch_loader,
        batch_loader_validation=None,
        model=model, experiments_path=tmp_dir.name, experiment_name='test',
        max_batches_per_epoch=4, max_epochs=2, patience=None,
        taskman_managed=False, use_gpu=False)

    return trainer
