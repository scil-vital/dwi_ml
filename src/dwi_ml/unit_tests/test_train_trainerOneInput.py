#!/usr/bin/env python
import logging
import os
import tempfile

import pytest

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.unit_tests.utils.data_and_models_for_tests import (
    create_test_batch_sampler, create_batch_loader, fetch_testing_data,
    ModelForTest, TrackingModelForTestWithPD)

SAVE_RESULT_SFT_NII = False
ref = None
batch_size = 50
batch_size_units = 'nb_streamlines'


@pytest.fixture(scope="session")
def experiments_path(tmp_path_factory):
    experiments_path = tmp_path_factory.mktemp("unit_tests")
    return str(experiments_path)


def test_trainer_and_models(experiments_path):

    # This unit test uses the test data.
    data_dir = fetch_testing_data()
    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')

    # Initializing dataset
    logging.info("Initializing dataset")
    dataset = MultiSubjectDataset(hdf5_filename, lazy=False)

    logging.info("Loading data (non-lazy)")
    dataset.load_data()

    # Initializing model 1 + associated batch sampler.
    logging.info("\n\n------------TESTING TEST MODEL # 1 : BASIC-------------")
    # Will be using streamline's group 0 and input group 0.
    model = ModelForTest(nb_features=dataset.nb_features[0])
    batch_sampler, batch_loader = _create_sampler_and_loader(dataset, model)

    # Start utils
    trainer = _create_trainer(batch_sampler, batch_loader, model,
                              experiments_path, 'test1')
    trainer.train_and_validate()

    # Initializing model 2
    logging.info("\n\n-----------TESTING TEST MODEL # 2: WITH PD AND "
                 "DIRECTION GETTER ------------")
    # Will be using streamline's group 0 and input group 0.

    model2 = TrackingModelForTestWithPD(nb_features=dataset.nb_features[0])
    batch_sampler, batch_loader = _create_sampler_and_loader(dataset, model)

    # Start utils
    trainer2 = _create_trainer(batch_sampler, batch_loader, model2,
                               experiments_path, 'test2')
    trainer2.train_and_validate()


def _create_sampler_and_loader(dataset, model):

    # Initialize batch sampler
    batch_sampler = create_test_batch_sampler(
        dataset, batch_size=batch_size,
        batch_size_units='nb_streamlines', log_level=logging.WARNING)

    batch_loader = create_batch_loader(dataset, model,
                                       log_level=logging.WARNING)

    return batch_sampler, batch_loader


def _create_trainer(batch_sampler, batch_loader, model, experiments_path,
                    experiment_name):

    trainer = DWIMLTrainerOneInput(
        batch_sampler=batch_sampler,
        batch_loader=batch_loader,
        model=model, experiments_path=str(experiments_path),
        experiment_name=experiment_name, log_level='DEBUG',
        max_batches_per_epoch_training=2,
        max_batches_per_epoch_validation=2, max_epochs=2, patience=None,
        use_gpu=False)
    # Note. toDo Test fails with nb_cpu_processes=1. Why??

    return trainer


if __name__ == '__main__':
    tmp_dir = tempfile.TemporaryDirectory()
    logging.getLogger().setLevel('DEBUG')
    test_trainer_and_models(tmp_dir.name)
