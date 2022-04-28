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
from dwi_ml.tests.expected_values import (
    TEST_EXPECTED_SUBJ_NAMES, TEST_EXPECTED_NB_STREAMLINES)
from dwi_ml.tests.utils import create_batch_sampler_loader, \
    create_test_batch_sampler, create_batch_loader

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
tmp_dir = tempfile.TemporaryDirectory()


logging.basicConfig(level=logging.DEBUG)
now = datetime.now().time()
millisecond = round(now.microsecond / 10000)
now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

# Change to true to allow debug mode: saves the mask for visual assessment.
# Requires a reference.
SAVE_RESULT_SFT_NII = False
ref = None  # Todo. Load from data fetcher


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
        # A.1) Batch size in terms of length in mm
        logging.debug('\n *** Test with batch size 1000 in terms of length_mm')
        batch_sampler = create_test_batch_sampler(
            dataset.training_set,
            batch_size=1000, chunk_size=100,
            batch_size_units='length_mm')
        _iterate_on_sampler_and_verify(
            batch_sampler, batch_size=1000, batch_size_units='length_mm')

        # A.2) Batch size in terms of number of streamlines
        total_nb_streamlines = TEST_EXPECTED_NB_STREAMLINES[0]
        for batch_size in [100, 25, total_nb_streamlines]:
            logging.debug('\n *** Test with batch size {} streamlines'
                          .format(batch_size))
            batch_sampler = create_test_batch_sampler(
                dataset.training_set,
                batch_size=batch_size,
                batch_size_units='nb_streamlines')
            batch_idx = _iterate_on_sampler_and_verify(
                batch_sampler, batch_size=batch_size,
                batch_size_units='nb_streamlines')

        # Using this last batch sampler:

        # B) Varying parameters for the batch loader.
        # B.1) With resampling
        for wait_for_gpu in [True, False]:
            logging.debug('\n *** Test with batch size 1000 + loading with '
                          'resample, noise, split, reverse, with '
                          'wait_for_gpu = {}'.format(wait_for_gpu))
            batch_loader = create_batch_loader(
                dataset.training_set, step_size=0.5, noise_size=0.2,
                noise_variability=0.1, split_ratio=0.5, reverse_ratio=0.5,
                wait_for_gpu=True)

            # Using last batch from batch sampler
            _load_directly_and_verify(
                batch_loader, batch_idx,
                expected_nb_streamlines=total_nb_streamlines, split_ratio=0.5)

            # Using torch's dataloader
            _load_from_torch_and_verify(
                dataset, batch_sampler, batch_loader,
                expected_nb_streamlines=total_nb_streamlines, split_ratio=0.5)

        # B.2) With compressing
        logging.debug('\n *** Test with batch size 1000 + loading with '
                      'compress')
        batch_loader = create_batch_loader(dataset.training_set, compress=True)
        _load_directly_and_verify(
            batch_loader, batch_idx,
            expected_nb_streamlines=total_nb_streamlines, split_ratio=0)


def _iterate_on_sampler_and_verify(
        batch_sampler, batch_size, batch_size_units):

    # Default variables
    nb_subjs = len(TEST_EXPECTED_SUBJ_NAMES)

    logging.debug('TESTING THE BATCH SAMPLER: \n'
                  'Iterating on the batch sampler directly, without actually '
                  'loading the batch.')
    batch_generator = batch_sampler.__iter__()

    # Init to avoid "batch referenced before assignment" in case generator
    # fails
    batch_idx = []

    # Loop on batches
    for batch_idx in batch_generator:
        subj0 = batch_idx[0]
        (subj0_id, subj0_streamlines) = subj0

        nb_streamlines_sampled = len(subj0_streamlines)
        if batch_size_units == 'nb_streamlines':
            logging.debug('Based on first subject, nb sampled streamlines per '
                          'subj was {} \n'
                          '(Batch size = {} streamlines)\n'
                          '(Result should be batch_size / nb_subjs ({}) = {})'
                          .format(nb_streamlines_sampled, batch_size, nb_subjs,
                                  batch_size / nb_subjs))

            assert nb_streamlines_sampled == batch_size / nb_subjs
        else:
            logging.debug('Based on first subject, nb sampled streamlines per '
                          'subj was {} \n'
                          '(Total length in mm unkown here. #todo)'
                          '(Batch size = {} in terms of length in mm)\n'
                          '(Result should be batch_size / nb_subjs ({}) = {})'
                          .format(nb_streamlines_sampled, batch_size, nb_subjs,
                                  batch_size / nb_subjs))
            break

    return batch_idx


def _load_directly_and_verify(batch_loader, batch_idx,
                              expected_nb_streamlines, split_ratio):
    # Debug mode for visual assessment :
    if SAVE_RESULT_SFT_NII:
        logging.debug("Debug mode. Saving input coordinates as mask. You can"
                      "open the mask and verify that they fit the streamlines")
        batch_streamlines, ids, inputs_tuple = batch_loader.load_batch(
            batch_idx, save_batch_input_mask=True)
        batch_input_masks, batch_inputs = inputs_tuple
        filename = os.path.join(str(tmp_dir), 'test_batch1_underlying_mask_' +
                                now_s + '.nii.gz')
        logging.debug("Saving subj 0's underlying coords mask to {}"
                      .format(filename))
        mask = batch_input_masks[0]
        ref_img = nib.load(ref)
        data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), ref_img)
        nib.save(data_nii, filename)
    else:
        logging.debug("Loading batch from our data loader")
        batch = batch_loader.load_batch(batch_idx)
        _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio)


def _load_from_torch_and_verify(
        dataset, batch_sampler, batch_loader, expected_nb_streamlines,
        split_ratio=0.):
    logging.debug('TESTING THE DATA LOADER: Using the batch sampler + loader '
                  'in a dataLoader')
    dataloader = DataLoader(dataset.training_set, batch_sampler=batch_sampler,
                            collate_fn=batch_loader.load_batch)

    # Iterating
    batch_sizes = []
    i = 0
    for i, batch in enumerate(dataloader):
        _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio)
        batch_sizes.append(len(batch[0]))

    # Total of batch (i.e. one epoch) should be all of the streamlines.
    # (if streamlines splitting not activated)
    if split_ratio == 0:
        logging.debug("Finished after i={} batches of average size {} with "
                      "total size {}"
                      .format(i + 1, np.mean(batch_sizes), sum(batch_sizes)))

        assert sum(batch_sizes) == TEST_EXPECTED_NB_STREAMLINES[0]

    # For debugging purposes: possibility to save the last batch's SFT.
    if SAVE_RESULT_SFT_NII :
        logging.info("Saving subj 0's tractogram {}"
                     .format('test_batch1_' + now_s))

        sft = StatefulTractogram(batch[0], reference=ref, space=Space.VOX,
                                 origin=Origin.TRACKVIS)
        filename = os.path.join(str(tmp_dir), 'test_batch_reverse_split_' +
                                now_s + '.trk')
        save_tractogram(sft, filename)


def _verify_loaded_batch(batch, expected_nb_streamlines, split_ratio):
    if split_ratio == 0:
        assert len(batch[0]) == expected_nb_streamlines, \
            "Error, the batch size should be maximum {} but we got {}" \
            .format(expected_nb_streamlines, len(batch[0]))
    else:
        assert len(batch[0]) == expected_nb_streamlines * 2, \
            "Error, the batch size should be maximum {} (*2, split) but " \
            "we got {}".format(expected_nb_streamlines, len(batch[0]))


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_batch_sampler_and_loader()
