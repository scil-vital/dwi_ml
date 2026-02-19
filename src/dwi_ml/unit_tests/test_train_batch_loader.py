#!/usr/bin/env python
import logging
import os

import numpy as np
from dipy.io.stateful_tractogram import set_sft_logger_level
from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.models.main_models import ModelWithOneInput
from dwi_ml.unit_tests.utils.expected_values import TEST_EXPECTED_NB_STREAMLINES
from dwi_ml.unit_tests.utils.data_and_models_for_tests import (
    create_test_batch_sampler, create_batch_loader, fetch_testing_data)

SPLIT_RATIO = 0.5


def test_batch_loader():
    data_dir = fetch_testing_data()

    logging.getLogger().setLevel(level=logging.INFO)

    batch_size = min(500, TEST_EXPECTED_NB_STREAMLINES[0])

    logging.info("Unit test: batch sampler iteration")
    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')

    for lazy in [False, True]:
        # Initialize dataset
        if lazy:
            logging.info('Initializing LAZY dataset...')
        else:
            logging.info('Initializing NON-LAZY dataset...')

        dataset = MultiSubjectDataset(hdf5_filename,
                                      lazy=False, log_level=logging.WARNING)
        dataset.load_data()
        # Faking a validation set
        dataset.validation_set = dataset.training_set

        # Will be using streamline's group 0 and input group 0.

        # Initialize batch sampler. Only using units 'nb_streamlines' here
        # for easier management. See test_batch_sampler for other uses of the
        # batch sampler.
        batch_size_per_subj = round(batch_size /
                                    dataset.training_set.nb_subjects)

        batch_id_sampler = create_test_batch_sampler(
            dataset, batch_size=batch_size,
            batch_size_units='nb_streamlines', log_level=logging.WARNING)
        batch_id_sampler.set_context('training')

        batch_id_generator = batch_id_sampler.__iter__()
        batch_idx_tuples = next(batch_id_generator)
        subj0, batch_indices = batch_idx_tuples[0]
        assert len(batch_indices) == batch_size_per_subj, \
            "Error, batch size was set to {} per subjects but the batch " \
            "sampler sampled {} streamlines." \
            .format(batch_size_per_subj, len(batch_indices))

        # Now testing.
        # 1) With resampling
        logging.info('*** Test with batch size {} + loading with '
                     'resample, noise, split, reverse.'.format(batch_size))
        model = ModelWithOneInput(experiment_name='test', step_size=0.5,
                                  nb_features=dataset.nb_features[0])
        batch_loader = create_batch_loader(
            dataset, model, noise_size=0.2, split_ratio=SPLIT_RATIO,
            reverse_ratio=0.5)
        batch_loader.set_context('training')

        # Using last batch from batch sampler
        _load_directly_and_verify(batch_loader, batch_idx_tuples)

        # Using torch's dataloader
        _load_from_torch_and_verify(dataset.training_set, batch_id_sampler,
                                    batch_loader)

        # 2) With compressing
        logging.info('*** Test with batch size {} + loading with compress'
                     .format(batch_size))
        model = ModelWithOneInput(experiment_name='test', compress_lines=True,
                                  nb_features=dataset.nb_features[0])
        batch_loader = create_batch_loader(dataset, model)
        batch_loader.set_context('training')
        _load_directly_and_verify(batch_loader, batch_idx_tuples,
                                  split_ratio=0)


def _load_directly_and_verify(batch_loader, batch_idx_tuples,
                              split_ratio=SPLIT_RATIO):
    expected_nb_streamlines = 0
    for s, idx in batch_idx_tuples:
        expected_nb_streamlines += len(idx)

    logging.info("    Loading batch directly from our load_batch method.")
    batch_idx_tuples = batch_loader.load_batch_streamlines(batch_idx_tuples)
    _verify_loaded_batch(batch_idx_tuples, expected_nb_streamlines,
                         split_ratio)


def _load_from_torch_and_verify(
        dataset, batch_sampler, batch_loader, split_ratio=SPLIT_RATIO):
    logging.info('TESTING THE DATA LOADER: Using the batch sampler + loader '
                 'in a dataLoader')

    dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                            collate_fn=batch_loader.load_batch_streamlines)

    batch_loader.set_context('training')
    # Iterating
    batch_sizes = []
    for i, batch in enumerate(dataloader):
        # expected_nb_streamlines = batch_sampler.batch_size
        # _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio)
        batch_sizes.append(len(batch[0]))

    # Total of batch (i.e. one epoch) should be all of the streamlines.
    # (if streamlines splitting not activated)
    if split_ratio == 0:
        logging.info("Finished after i={} batches of average size {} with "
                     "total size {}"
                     .format(i + 1, np.mean(batch_sizes), sum(batch_sizes)))

        assert sum(batch_sizes) == TEST_EXPECTED_NB_STREAMLINES[0]

    # Confirm that batch loader is using the right context
    # We cannot really assert anything here but during logs, verify
    # that the noise is indeed 0, contrary to training.
    logging.info("*** Test on validation")
    batch_loader.set_context('validation')
    batch_sizes = []
    for i, batch in enumerate(dataloader):
        # expected_nb_streamlines = batch_sampler.batch_size
        # _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio)
        batch_sizes.append(len(batch[0]))


def _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio):
    if split_ratio == 0:
        logging.info("   After loading the data, batch size should be {} "
                     "(batch size in nb_streamlines): {}"
                     .format(expected_nb_streamlines, len(batch[0])))
        assert len(batch[0]) == expected_nb_streamlines
    else:
        nb_streamlines_not_split = expected_nb_streamlines * (1 - split_ratio)
        nb_streamlines_split = expected_nb_streamlines * split_ratio
        real_expected_nb = nb_streamlines_not_split * 1 + \
            nb_streamlines_split * 2
        logging.info("   After loading the data, batch size should be {} "
                     "(batch size in nb_streamlines; {} + accounting for "
                     "split ratio): {}"
                     .format(real_expected_nb, expected_nb_streamlines,
                             len(batch[0])))
        assert len(batch[0]) == real_expected_nb


if __name__ == '__main__':
    logging.getLogger().setLevel(level='INFO')
    # Silencing SFT's logger if our logging is in DEBUG mode, because it
    # typically produces a lot of outputs!
    set_sft_logger_level('WARNING')
    test_batch_loader()
