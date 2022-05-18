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

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.tests.utils import create_test_batch_sampler, create_batch_loader

# fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
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


def save_loaded_batch_for_visual_assessment():
    logging.info("Unit test: batch sampler iteration")

    # os.chdir(os.path.expanduser(tmp_dir.name))
    # hdf5_filename = os.path.join(get_home(), 'dwiml', 'hdf5_file.hdf5')
    home = os.path.expanduser("~")
    hdf5_filename = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')

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
        _load_directly_and_verify(
            batch_loader, batch_idx_tuples, split_ratio=0.5)

        # 2) With compressing
        logging.info('*** Test with batch size {} + loading with compress'
                     .format(batch_size))
        batch_loader = create_batch_loader(dataset.training_set, compress=True)
        _load_directly_and_verify(
            batch_loader, batch_idx_tuples, split_ratio=0)


def _load_directly_and_verify(batch_loader, batch_idx_tuples, split_ratio):
    expected_nb_streamlines = 0
    for s, idx in batch_idx_tuples:
        expected_nb_streamlines += len(idx)

    # Saving input coordinates as mask. You can "open the mask and verify that
    # they fit the streamlines.
    batch_streamlines, ids, inputs_tuple = batch_loader.load_batch(
        batch_idx_tuples, save_batch_input_mask=True)
    batch_input_masks, batch_inputs = inputs_tuple
    filename = os.path.join(str(tmp_dir), 'test_batch1_underlying_mask_' +
                            now_s + '.nii.gz')
    logging.info("Saving subj 0's underlying coords mask to {}"
                 .format(filename))
    mask = batch_input_masks[0]
    ref_img = nib.load(ref)
    data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), ref_img)
    nib.save(data_nii, filename)

    # Save the last batch's SFT.
    logging.info("Saving subj 0's tractogram {}"
                 .format('test_batch1_' + now_s))

    sft = StatefulTractogram(batch_streamlines, reference=ref, space=Space.VOX,
                             origin=Origin.TRACKVIS)
    filename = os.path.join(str(tmp_dir), 'test_batch_reverse_split_' +
                            now_s + '.trk')
    save_tractogram(sft, filename)


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    save_loaded_batch_for_visual_assessment()
