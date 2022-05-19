#!/usr/bin/env python
from datetime import datetime
import logging
import os
import tempfile

import numpy as np
from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.tests.expected_values import TEST_EXPECTED_NB_STREAMLINES
from dwi_ml.tests.utils import (
    create_test_batch_sampler, create_batch_loader, fetch_testing_data)

data_dir = fetch_testing_data()
tmp_dir = tempfile.TemporaryDirectory()

logging.basicConfig(level=logging.INFO)
now = datetime.now().time()
millisecond = round(now.microsecond / 10000)
now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

# Change to true to allow debug mode: saves the mask for visual assessment.
# Requires a reference.
ref = None  # Todo. Load from data fetcher
batch_size = 500  # Testing only one value here.
wait_for_gpu = False  # Testing both True and False is heavier...


def test_batch_loader():
    logging.info("Unit test: batch sampler iteration")
    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')

    for lazy in [False, True]:
        # Initialize dataset
        if lazy:
            logging.info('Initializing LAZY dataset...')
        else:
            logging.info('Initializing NON-LAZY dataset...')

        dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=False,
                                      lazy=False, log_level=logging.WARNING)
        dataset.load_data()

        # Initialize batch sampler. Only using units 'nb_streamlines' here
        # for easier management. See test_batch_sampler for other uses of the
        # batch sampler.
        batch_size_per_subj = round(batch_size /
                                    dataset.training_set.nb_subjects)

        batch_sampler = create_test_batch_sampler(
            dataset.training_set, batch_size=batch_size,
            batch_size_units='nb_streamlines', log_level=logging.WARNING)
        batch_generator = batch_sampler.__iter__()
        batch_idx_tuples = next(batch_generator)
        subj0, batch_indices = batch_idx_tuples[0]
        assert len(batch_indices) == batch_size_per_subj, \
            "Error, batch size was set to {} per subjects but the batch " \
            "sampler sampled {} streamlines." \
            .format(batch_size_per_subj, len(batch_indices))

        # Now testing.
        # 1) With resampling
        logging.info('*** Test with batch size {} + loading with '
                     'resample, noise, split, reverse, with '
                     'wait_for_gpu = {}'.format(batch_size, wait_for_gpu))
        batch_loader = create_batch_loader(
            dataset.training_set, step_size=0.5, noise_size=0.2,
            noise_variability=0.1, split_ratio=0.5, reverse_ratio=0.5,
            wait_for_gpu=True)

        # Using last batch from batch sampler
        _load_directly_and_verify(batch_loader, batch_idx_tuples,
                                  split_ratio=0.5)

        # Using torch's dataloader
        _load_from_torch_and_verify(dataset, batch_sampler, batch_loader,
                                    split_ratio=0.5)

        # 2) With compressing
        logging.info('*** Test with batch size {} + loading with compress'
                     .format(batch_size))
        batch_loader = create_batch_loader(dataset.training_set, compress=True)
        _load_directly_and_verify(batch_loader, batch_idx_tuples,
                                  split_ratio=0)


def _load_directly_and_verify(batch_loader, batch_idx_tuples, split_ratio):
    expected_nb_streamlines = 0
    for s, idx in batch_idx_tuples:
        expected_nb_streamlines += len(idx)

    logging.info("    Loading batch directly from our load_batch method.")
    batch_idx_tuples = batch_loader.load_batch(batch_idx_tuples)
    _verify_loaded_batch(batch_idx_tuples, expected_nb_streamlines,
                         split_ratio)


def _load_from_torch_and_verify(
        dataset, batch_sampler, batch_loader, split_ratio=0.):
    logging.info('TESTING THE DATA LOADER: Using the batch sampler + loader '
                 'in a dataLoader')

    dataloader = DataLoader(dataset.training_set, batch_sampler=batch_sampler,
                            collate_fn=batch_loader.load_batch)

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
    logging.basicConfig(level='DEBUG')
    test_batch_loader()
