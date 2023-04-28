#!/usr/bin/env python
# -*- coding: utf-8 -*-

from datetime import datetime
import logging
import os
import tempfile

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.models.main_models import MainModelOneInput
from dwi_ml.unit_tests.utils.data_and_models_for_tests import (
    create_test_batch_sampler, create_batch_loader, fetch_testing_data,
    ModelForTest)

tmp_dir = tempfile.TemporaryDirectory()
batch_size = 500  # Testing only one value here.
results_folder = './'


def save_loaded_batch_for_visual_assessment(dataset, ref):

    # Initialize batch sampler using units 'nb_streamlines'.
    batch_sampler = create_test_batch_sampler(
        dataset, batch_size=batch_size,
        batch_size_units='nb_streamlines', log_level=logging.WARNING)
    batch_sampler.set_context('training')

    # Creating a batch generator
    batch_generator = batch_sampler.__iter__()
    batch_idx_tuples = next(batch_generator)  # tuples of subj, batch_indices

    # Creating a model
    model = MainModelOneInput(experiment_name='test')

    # Now testing.
    # 1) With resampling + split + reverse
    logging.info("*********")
    logging.info("Split + reverse:")
    batch_loader = create_batch_loader(
        dataset, model, noise_size=0.2, split_ratio=1, reverse_ratio=0.5)
    batch_loader.set_context('training')
    _load_directly_and_verify(batch_loader, batch_idx_tuples, ref, 'split')

    # 2) With compressing
    logging.info("*********")
    logging.info("Compressed:")
    batch_loader = create_batch_loader(dataset, model)
    batch_loader.set_context('training')
    _load_directly_and_verify(batch_loader, batch_idx_tuples, ref, 'compress')

    # 3) With neighborhood
    logging.info("*********")
    logging.info("Neighborhood:")
    # Using ModelForTest with both parents OneInput and Neighborhood.
    model = ModelForTest(experiment_name='test',
                         neighborhood_type='axes',
                         neighborhood_radius=[1, 2])
    batch_loader = create_batch_loader(dataset, model)
    batch_loader.set_context('training')
    _load_directly_and_verify(batch_loader, batch_idx_tuples, ref, 'neighb')


def _load_directly_and_verify(batch_loader, batch_idx_tuples, ref, suffix):
    # Prepare suffix for filename
    now = datetime.now().time()
    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)
    suffix += '_' + now_s

    # Load batch
    batch_streamlines, ids = batch_loader.load_batch_streamlines(
        batch_idx_tuples)

    inputs_tuple = batch_loader.load_batch_inputs(
        batch_streamlines, ids, save_batch_input_mask=True)

    # Saving input coordinates as mask. You can open the mask and verify that
    # they fit the streamlines.
    mask, inputs = inputs_tuple
    mask = np.asarray(mask, dtype=bool)
    filename = os.path.join(results_folder, 'test_batch1_underlying_mask_' +
                            suffix + '.nii.gz')
    logging.info("Saving subj 0's underlying coords mask to {}"
                 .format(filename))
    data_nii = nib.Nifti1Image(mask, ref.affine, ref.header)
    nib.save(data_nii, filename)

    # Save the last batch's SFT.
    logging.info("Saving subj's tractogram {}".format('test_batch1_' + suffix))
    sft = StatefulTractogram(batch_streamlines, reference=ref, space=Space.VOX,
                             origin=Origin('corner'))

    sft.data_per_streamline = {"seeds": [s[0] for s in batch_streamlines]}
    filename = os.path.join(results_folder, 'test_batch_' + suffix + '.trk')
    save_tractogram(sft, filename)


def main():
    logging.getLogger().setLevel(level='DEBUG')

    data_dir = fetch_testing_data()
    ref = os.path.join(data_dir, 'dwi_ml_ready', 'subjX', 'dwi', 'fa.nii.gz')
    hdf5_filename = os.path.join(data_dir, 'hdf5_file.hdf5')

    ref_img = nib.load(ref)
    nib.save(ref_img, os.path.join(results_folder, 'test_ref.nii.gz'))

    for lazy in [False, True]:
        # Initialize dataset
        if lazy:
            logging.info('Initializing LAZY dataset...')
        else:
            logging.info('Initializing NON-LAZY dataset...')

        dataset = MultiSubjectDataset(hdf5_filename,
                                      lazy=False, log_level=logging.WARNING)
        dataset.load_data()

        save_loaded_batch_for_visual_assessment(dataset, ref_img)


if __name__ == '__main__':
    main()
