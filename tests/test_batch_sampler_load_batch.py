#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from os import path

import numpy as np
from torch.utils.data.dataloader import DataLoader

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
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects.')
    return p.parse_args()


def test_batch_loading_no_computations(fake_dataset, batch_size, step_size):
    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = np.random.RandomState(1234)
    batch_sampler = BatchSequencesSamplerOneInputVolume(
        fake_dataset, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=True)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        break

    batch_streamlines, batch_ids = batch_sampler.load_batch(batch)

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))


def test_batch_loading_computations(fake_dataset, batch_size, step_size):
    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = np.random.RandomState(1234)
    batch_sampler = BatchSequencesSamplerOneInputVolume(
        fake_dataset, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=False)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        break

    packed_inputs, packed_directions = batch_sampler.load_batch(batch)

    print('Nb loaded processed batch streamlines: packed inputs: {}. '
          'packed_directions: {}, creating {} streamlines'
          .format(len(packed_inputs.data), len(packed_directions.data),
                  len(packed_directions.sorted_indices)))

    print('inputs size: {} contains Nan because the mask I used was wm_mask, '
          'seeding was interface. nanmean: {} '
          'Remember to deal with neighborhood being outside of mask too.'
          .format(len(packed_inputs.data), np.nanmean(packed_inputs.data)))

def test_non_lazy():
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()

    print('=============================Test with batch size 10000 + resample')
    test_batch_loading_no_computations(fake_dataset, 10000, 0.5)

    print('=============================Test with batch size 10000 + resample '
          '+ do cpu computations')
    test_batch_loading_computations(fake_dataset, 10000, 0.5)


def test_lazy():
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = LazyMultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()

    print('=============================Test with batch size 10000')
    test_batch_loading(fake_dataset, 10000, 0.5)


if __name__ == '__main__':
    args = parse_args()

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    logging.basicConfig(level='DEBUG')

    test_non_lazy()
    print('\n\n')
    #test_lazy()
