#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from datetime import datetime
from os import path

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import (StatefulTractogram,
                                         set_sft_logger_level)
from dipy.io.streamline import (save_tractogram, Space)

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)
from dwi_ml.model.batch_samplers import (BatchSequencesSamplerOneInputVolume)


def parse_args():
    """
    Convert "raw" streamlines from a hdf5 to torch data or to PackedSequences
    via our MultiSubjectDataset class. Test some properties.
    The MultiSubjectDataset is used during training, in the trainer_abstract.
    """
    p = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                formatter_class=RawTextHelpFormatter)
    p.add_argument('hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain only one '
                        'training subj for this test to be able to save '
                        'output streamlines')
    p.add_argument('ref',
                   help='Ref MRI volume to save the streamlines')
    p.add_argument('saving_path',
                   help='Path for saving tests (.trk, .nii)')
    return p.parse_args()


def test_batch_loading_no_computations(now, fake_dataset, batch_size,
                                       step_size):
    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = np.random.RandomState(now.minute * 100 + now.second)
    batch_sampler = BatchSequencesSamplerOneInputVolume(
        fake_dataset, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=True)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        print('Batch # 1 streamline #1: {}'.format(batch[0][0]))
        break

    batch_streamlines, batch_ids = batch_sampler.load_batch(batch)

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    return batch_streamlines


def test_batch_loading_computations(now, fake_dataset, batch_size, step_size,
                                    ref, affine, header, saving_path):
    logging.root.setLevel('DEBUG')
    set_sft_logger_level('WARNING')

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = np.random.RandomState(now.minute * 100 + now.second)
    batch_sampler = BatchSequencesSamplerOneInputVolume(
        fake_dataset, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=False)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        print('Batch # 1 streamline #1: {}'.format(batch[0][0]))
        break

    batch_streamlines, batch_input_masks = batch_sampler.load_batch(
        batch, save_batch_input_masks=True)

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    now_s = str(now.minute * 100 + now.second)

    print("Saving subj 0's tractogram {}".format('test_batch1_' + now_s))
    sft = StatefulTractogram(batch_streamlines, ref, space=Space.VOX)
    save_tractogram(sft, saving_path + '/test_batch1_' + now_s + '.trk')

    print("Saving subj 0's underlying coords mask: {}"
          .format('test_batch1_underlying_mask_' + now_s ))
    mask = batch_input_masks[0]
    data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), affine, header)
    nib.save(data_nii, saving_path + '/test_batch1_underlying_mask_' + now_s +
             '.nii.gz')


def test_non_lazy(now, ref, affine, header, saving_path):
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()

    print('\n===========================Test with batch size 10000 + resample')
    test_batch_loading_no_computations(now, fake_dataset, 10000, 0.5)

    print('\n===========================Test with batch size 10000 + resample '
          '+ do cpu computations')
    test_batch_loading_computations(now, fake_dataset, 10000, 0.5, ref,
                                    affine, header, saving_path)


def test_lazy(n, ref, affine, header, saving_path):
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = LazyMultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()

    print('\n===========================Test with batch size 10000 + resample')
    test_batch_loading_computations(n, fake_dataset, 10000, 0.5)


if __name__ == '__main__':
    args = parse_args()
    logging.basicConfig(level='INFO')

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    img = nib.load(args.ref)
    affine = img.affine
    header = img.header
    print('Saving path = {}'.format(args.saving_path))

    now = datetime.now().time()
    set_sft_logger_level('WARNING')

    test_non_lazy(now, args.ref, affine, header, args.saving_path)
    print('\n\n')
    # test_lazy(now, args.ref, affine, header, args.saving_path)
