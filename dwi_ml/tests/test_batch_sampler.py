#!/usr/bin/env python
import logging
import os

from dipy.tracking.metrics import length

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_SUBJ_NAMES, TEST_EXPECTED_NB_STREAMLINES)
from dwi_ml.tests.utils import create_test_batch_sampler, fetch_testing_data


def test_batch_sampler():
    data_dir = fetch_testing_data()

    # Change to true to allow debug mode: saves the mask for visual assessment.
    # Requires a reference.

    logging.debug("Unit test: batch sampler iteration")

    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')

    for lazy in [False, True]:
        # Initialize dataset
        if lazy:
            logging.debug('Initializing LAZY dataset...')
        else:
            logging.debug('Initializing NON-LAZY dataset...')

        dataset = MultiSubjectDataset(hdf5_filename, lazy=False,
                                      log_level=logging.WARNING)
        dataset.load_data()

        # 1) Batch size in terms of length in mm
        batch_size = 5000
        logging.debug('*** Test with batch size {} in terms of length_mm'
                      .format(batch_size))
        batch_sampler = create_test_batch_sampler(
            dataset, batch_size=batch_size, chunk_size=100,
            batch_size_units='length_mm', log_level=logging.DEBUG)
        batch_sampler.set_context('training')

        iterate_on_sampler_and_verify(
            batch_sampler, batch_size=batch_size, batch_size_units='length_mm')

        # 2) Batch size in terms of number of streamlines
        total_nb_streamlines = TEST_EXPECTED_NB_STREAMLINES[0]
        for batch_size in [100, 25, total_nb_streamlines]:
            logging.debug('*** Test with batch size {} streamlines'
                          .format(batch_size))
            batch_sampler = create_test_batch_sampler(
                dataset, batch_size=batch_size,
                batch_size_units='nb_streamlines')
            batch_sampler.set_context('training')
            iterate_on_sampler_and_verify(
                batch_sampler, batch_size=batch_size,
                batch_size_units='nb_streamlines')


def iterate_on_sampler_and_verify(
        batch_sampler, batch_size, batch_size_units):
    # Default variables
    nb_subjs = len(TEST_EXPECTED_SUBJ_NAMES)

    logging.debug('    Iterating on the batch sampler directly, without '
                  'actually loading the batch.')
    batch_generator = batch_sampler.__iter__()

    # Init to avoid "batch referenced before assignment" in case generator
    # fails

    # Loop on batches
    for batch_idx in batch_generator:
        subj0 = batch_idx[0]
        (subj0_id, subj0_streamlines_idx) = subj0

        nb_streamlines_sampled = len(subj0_streamlines_idx)
        if batch_size_units == 'nb_streamlines':
            logging.debug(
                '     Based on first subject, nb sampled streamlines per subj '
                'was {} \n'
                '     (Batch size should be {} streamlines, result should be '
                'batch_size / nb_subjs ({}) = {})\n'
                .format(nb_streamlines_sampled, batch_size, nb_subjs,
                        batch_size / nb_subjs))

            assert nb_streamlines_sampled == batch_size / nb_subjs
        else:
            subj0 = batch_sampler.context_subset.subjs_data_list[0]
            sub0_sft = subj0.sft_data_list[0].as_sft(subj0_streamlines_idx)
            sub0_sft.to_rasmm()
            lengths = [length(s) for s in sub0_sft.streamlines]
            computed_size = sum(lengths)

            logging.debug(
                '    Based on first subject, nb sampled streamlines per subj '
                'was {} for a total size of {}\n'
                '    (Batch size should be {} in terms of length in mm, '
                'result should be batch_size / nb_subjs ({}) = {})\n'
                .format(nb_streamlines_sampled, computed_size,
                        batch_size, nb_subjs, batch_size / nb_subjs))

            allowed_error = 200  # Usually, biggest streamline length is 200mm
            assert batch_size - computed_size < allowed_error
        break


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_batch_sampler()

