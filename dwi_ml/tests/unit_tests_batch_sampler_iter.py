#!/usr/bin/env python
import logging
import os
import tempfile

import numpy as np
import pytest
from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict
from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.tests.expected_values import *

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
# tmp_dir = tempfile.TemporaryDirectory()


logging.basicConfig(level=logging.DEBUG)


def test_multisubjectdataset():
    logging.debug("\n"
                  "Unit test: batch sampler iteration\n"
                  "------------------------")

    # os.chdir(os.path.expanduser(tmp_dir.name))
    # hdf5_filename = os.path.join(get_home(), 'dwiml', 'hdf5_file.hdf5')
    home = os.path.expanduser("~")
    hdf5_filename = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')

    for lazy in [False, True]:
        # Initialize dataset
        if lazy:
            logging.debug('Initializing LAZY dataset...')
        else:
            logging.debug('Initializing NON-LAZY dataset...')
            
        dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=False,
                                      lazy=False)
        dataset.load_data()

        total_nb_streamlines = TEST_EXPECTED_NB_STREAMLINES[0]
        for batch_size in [100, 25, total_nb_streamlines]:
            logging.debug('\n================= Test with batch size {} '
                          'streamlines'.format(batch_size))
            _create_sampler_and_iterate(dataset, batch_size=batch_size,
                                        batch_size_units='nb_streamlines')

        logging.debug('\n=================== Test with batch size 1000 + '
                      'resample')
        _create_sampler_and_iterate(dataset, batch_size=1000,
                                    batch_size_units='nb_streamlines',
                                    step_size=0.5)

        logging.debug('\n=================== Test with batch size 1000 + '
                      'compress')
        _create_sampler_and_iterate(dataset, batch_size=1000,
                                    batch_size_units='nb_streamlines',
                                    compress=True)

        logging.debug('\n=================== Test with batch size 1000 in '
                      'terms of length_mm')
        _create_sampler_and_iterate(dataset, batch_size=1000, chunk_size=100,
                                    batch_size_units='length_mm')


def _create_sampler_and_iterate(dataset, batch_size, batch_size_units,
                                chunk_size=None, step_size=None,
                                compress=False):
    # Initialize batch sampler
    logging.debug('\nInitializing sampler...')
    training_set = dataset.training_set

    batch_sampler = DWIMLBatchSampler(
        training_set, TEST_EXPECTED_STREAMLINE_GROUPS[0],
        batch_size=batch_size, batch_size_units=batch_size_units,
        nb_streamlines_per_chunk=chunk_size, rng=1234,
        nb_subjects_per_batch=1, cycles=1)

    batch_loader = BatchLoaderOneInput(
        training_set, TEST_EXPECTED_STREAMLINE_GROUPS[0], rng=1234,
        compress=compress, step_size=step_size, split_ratio=0,
        noise_gaussian_size=0, noise_gaussian_variability=0,
        reverse_ratio=0,
        input_group_name=TEST_EXPECTED_VOLUME_GROUPS[0], wait_for_gpu=True,
        neighborhood_points=None)

    # Use it in the dataloader
    # it says error, that the collate_fn should receive a list, but it is
    # supposed to work with dict too.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/1918
    logging.debug('TEST 1: Using it in a dataLoader')
    dataloader = DataLoader(training_set, batch_sampler=batch_sampler,
                            collate_fn=batch_loader.load_batch)

    batch_sizes = []
    for i, batch in enumerate(dataloader):
        # logging.debug("Nb streamlines in batch: {}".format(len(batch[0])))
        assert len(batch[0]) <= batch_size
        batch_sizes.append(len(batch[0]))

    logging.debug("Finished after i={} batches of average size {} with total "
                  "size {}".format(i+1, np.mean(batch_sizes),
                                   sum(batch_sizes)))
    assert sum(batch_sizes) == TEST_EXPECTED_NB_STREAMLINES[0]

    logging.debug('TEST 2: Iterating on the batch sampler directly')
    batch_generator = batch_sampler.__iter__()

    # Loop on batches
    nb_subjs = len(TEST_EXPECTED_SUBJ_NAMES)
    for batch in batch_generator:
        subj0 = batch[0]
        (subj0_id, subj0_streamlines) = subj0

        if batch_size_units == 'nb_streamlines':
            logging.debug('Based on first subject, nb sampled streamlines per '
                          'subj was {} (Should be {} / {} = {})'
                          .format(len(subj0_streamlines), batch_size, nb_subjs,
                                  batch_size / nb_subjs))

            assert len(subj0_streamlines) == batch_size / nb_subjs
        break


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_multisubjectdataset()
