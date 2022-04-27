#!/usr/bin/env python
from datetime import datetime
import logging
import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram
from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict
from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.batch_samplers import DWIMLBatchSampler
from dwi_ml.training.batch_loaders import BatchLoaderOneInput
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_SUBJ_NAMES, TEST_EXPECTED_STREAMLINE_GROUPS,
    TEST_EXPECTED_VOLUME_GROUPS, TEST_EXPECTED_NB_STREAMLINES)

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()


logging.basicConfig(level=logging.DEBUG)
SAVE_RESULT_SFT_NII = False
ref = None


def test_batch_sampler_and_loader():
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

        # A) Varying parameters for the batch sampler
        total_nb_streamlines = TEST_EXPECTED_NB_STREAMLINES[0]
        for batch_size in [100, 25, total_nb_streamlines]:
            logging.debug('\n================= Test with batch size {} '
                          'streamlines'.format(batch_size))
            _create_sampler_and_iterate(dataset, batch_size=batch_size,
                                        batch_size_units='nb_streamlines')

        logging.debug('\n=================== Test with batch size 1000 in '
                      'terms of length_mm')
        _create_sampler_and_iterate(dataset, batch_size=1000, chunk_size=100,
                                    batch_size_units='length_mm')

        # B) Varying parameters for the batch loader.
        for wait_for_gpu in [True, False]:
            logging.debug('\n=================== Test with batch size 1000 + '
                          'loading with resample, noise, split, reverse, '
                          'with wait_for_gpu = {}'.format(wait_for_gpu))
            _create_sampler_and_iterate(dataset, batch_size=1000,
                                        batch_size_units='nb_streamlines',
                                        step_size=0.5, noise_size=0.2,
                                        noise_variability=0.1, split_ratio=0.5,
                                        reverse_ratio=0.5)

        logging.debug('\n=================== Test with batch size 1000 + '
                      'loading with compress')
        _create_sampler_and_iterate(dataset, batch_size=1000,
                                    batch_size_units='nb_streamlines',
                                    compress=True)


def _create_sampler_and_iterate(
        dataset, batch_size, batch_size_units, chunk_size=None,
        step_size=None, compress=False, noise_size=0., noise_variability=0.,
        split_ratio= 0., reverse_ratio=0., wait_for_gpu=True):
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
        compress=compress, step_size=step_size, split_ratio=split_ratio,
        noise_gaussian_size=noise_size,
        noise_gaussian_variability=noise_variability,
        reverse_ratio=reverse_ratio,
        input_group_name=TEST_EXPECTED_VOLUME_GROUPS[0],
        neighborhood_points=None, wait_for_gpu=wait_for_gpu)

    now = datetime.now().time()
    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

    # -------------
    # Test 1 : Batch sampler
    # -------------
    logging.debug('TESTING THE BATCH SAMPLER: \n'
                  'Iterating on the batch sampler directly, without actually '
                  'loading the batch.')
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

    if SAVE_RESULT_SFT_NII:
        logging.debug("Debug mode. Saving input coordinates as mask.")
        batch_streamlines, ids, inputs_tuple = batch_loader.load_batch(
            batch, save_batch_input_mask=True)
        batch_input_masks, batch_inputs = inputs_tuple
        filename = os.path.join(str(tmp_dir), 'test_batch1_underlying_mask_' +
                                now_s + '.nii.gz')
        logging.debug("Saving subj 0's underlying coords mask to {}"
                      .format(filename))
        mask = batch_input_masks[0]
        ref_img = nib.load(ref)
        data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), ref_img)
        nib.save(data_nii, filename)

    # -------------
    # Test 2: Batch loader
    # -------------
    logging.debug('TESTING THE DATA LOADER: Using the batch sampler + loader '
                  'in a dataLoader')
    dataloader = DataLoader(training_set, batch_sampler=batch_sampler,
                            collate_fn=batch_loader.load_batch)

    # Iterating
    batch_sizes = []
    for i, batch in enumerate(dataloader):
        if split_ratio == 0:
            assert len(batch[0]) <= batch_size, \
                "Error, the batch size should be maximum {} but we got {}" \
                .format(batch_size, len(batch[0]))
        else:
            assert len(batch[0]) <= batch_size * 2, \
                "Error, the batch size should be maximum {} (*2, split) but " \
                "we got {}".format(batch_size, len(batch[0]))
        batch_sizes.append(len(batch[0]))

    # Total of batch (i.e. one epoch) should be all of the streamlines.
    # (if streamlines splitting not activated)
    if split_ratio == 0:
        logging.debug("Finished after i={} batches of average size {} with "
                      "total size {}"
                      .format(i + 1, np.mean(batch_sizes), sum(batch_sizes)))

        if batch_size_units == 'nb_streamlines':
            assert sum(batch_sizes) == TEST_EXPECTED_NB_STREAMLINES[0]
        else:
            assert sum(batch_sizes) <= TEST_EXPECTED_NB_STREAMLINES[0]

    # For debugging purposes: possibility to save the last batch's SFT.
    if SAVE_RESULT_SFT_NII :
        logging.info("Saving subj 0's tractogram {}"
                     .format('test_batch1_' + now_s))

        sft = StatefulTractogram(batch[0], reference=ref, space=Space.VOX,
                                 origin=Origin.TRACKVIS)
        filename = os.path.join(str(tmp_dir), 'test_batch_reverse_split_' +
                                now_s + '.trk')
        save_tractogram(sft, filename)


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_batch_sampler_and_loader()
