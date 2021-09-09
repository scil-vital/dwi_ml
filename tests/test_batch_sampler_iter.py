#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from os import path

from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)
from dwi_ml.model.batch_samplers import (BatchStreamlinesSampler1IPV)


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


def test_sampler(fake_dataset, batch_size, step_size):
    # Initialize batch sampler
    print('Initializing sampler...')
    batch_sampler = BatchStreamlinesSampler1IPV(
        fake_dataset, 'streamlines', 'input', batch_size, 1234,
        step_size=step_size)

    # Use it in the dataloader
    # it says error, that the collate_fn should receive a list, but it is
    # supposed to work with dict too.
    # See https://github.com/PyTorchLightning/pytorch-lightning/issues/1918
    print('Initializing DataLoader')
    dataloader = DataLoader(fake_dataset, batch_sampler=batch_sampler,
                            collate_fn=batch_sampler.load_batch)
    print(dataloader)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches
    i = 0
    total_sampled_ids_sub0 = []
    for batch in batch_generator:
        subj0 = batch[0]
        (subj0_id, subj0_streamlines) = subj0
        print('Batch # {}: nb sampled streamlines subj 0 was {}'
              .format(i, len(subj0_streamlines)))
        total_sampled_ids_sub0.append(batch[0])
        i = i + 1
        if i > 3:
            break


def test_non_lazy():
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, 'training_subjs')
    fake_dataset.load_data()

    print('\n=============================Test with batch size 1000')
    test_sampler(fake_dataset, 1000, None)

    print('\n=============================Test with batch size 10000')
    test_sampler(fake_dataset, 10000, None)

    print('\n===========================Test with batch size 10000 + resample')
    test_sampler(fake_dataset, 10000, 0.5)


def test_lazy():
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    fake_dataset = LazyMultiSubjectDataset(args.hdf5_filename,
                                           'training_subjs')
    fake_dataset.load_data()

    print('\n=============================Test with batch size 1000')
    test_sampler(fake_dataset, 1000, None)
    print('\n=============================Test with batch size 10000')
    test_sampler(fake_dataset, 10000, None)

    print('\n===========================Test with batch size 10000 + resample')
    test_sampler(fake_dataset, 10000, 0.5)


if __name__ == '__main__':
    args = parse_args()

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    logging.basicConfig(level='DEBUG')

    test_non_lazy()
    print('\n\n')
    test_lazy()
