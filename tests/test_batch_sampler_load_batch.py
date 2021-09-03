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

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
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
                   help='Path to the .hdf5 dataset. Should contain only one '
                        'training subj for this test to be able to save '
                        'output streamlines')
    p.add_argument('ref',
                   help='Ref MRI volume to save the streamlines')
    p.add_argument('saving_path',
                   help='Path for saving tests (.trk, .nii)')
    return p.parse_args()


def test_batch_loading_no_computations(
        fake_dataset, batch_size, step_size, ref, saving_path,
        split_ratio: float = 0, noise_size: float = 0,
        noise_variability: float = 0, reverse_ratio: float = 0):

    now = datetime.now().time()
    logging.root.setLevel('DEBUG')

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second
    batch_sampler = BatchStreamlinesSampler1IPV(
        training_set, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=True,
        split_streamlines_ratio=split_ratio,
        noise_gaussian_size=noise_size,
        noise_gaussian_variability=noise_variability,
        reverse_streamlines_ratio=reverse_ratio)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        break

    batch_streamlines, _ = batch_sampler.load_batch(batch)

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

    print("Saving subj 0's tractogram {}".format('test_batch1_' + now_s))
    sft = StatefulTractogram(batch_streamlines, ref, space=Space.VOX)
    save_tractogram(sft, saving_path + '/test_batch1_' + now_s + '.trk')

    return batch_streamlines


def test_batch_loading_computations(fake_dataset, batch_size, step_size,
                                    ref, affine, header, saving_path,
                                    neighborhood_type=None,
                                    neighborhood_radius=None):
    set_sft_logger_level('WARNING')
    logging.root.setLevel('DEBUG')
    now = datetime.now().time()

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second

    batch_sampler = BatchStreamlinesSampler1IPV(
        training_set, 'streamlines', 'input', batch_size, rng_seed,
        step_size=step_size, avoid_cpu_computations=False,
        neighborhood_type=neighborhood_type,
        neighborhood_radius_vox=neighborhood_radius,
        nb_previous_dirs=2)

    batch_generator = batch_sampler.__iter__()

    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        print("Batch # 1 streamline #1's id: {}".format(batch[0][0]))
        break

    print("\n\n TEST 1: Debug mode. Saving input coordinates as mask.")
    inputs, directions, previous_dirs = batch_sampler.load_batch(
        batch, save_batch_input_mask=True)

    batch_streamlines, batch_input_masks, batch_inputs = inputs

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

    print("Saving subj 0's tractogram {}".format('test_batch1_' + now_s))
    sft = StatefulTractogram(batch_streamlines, ref, space=Space.VOX)
    save_tractogram(sft, saving_path + '/test_batch1_' + now_s + '.trk')

    print("Saving subj 0's underlying coords mask: {}"
          .format('test_batch1_underlying_mask_' + now_s))
    mask = batch_input_masks[0]
    data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), affine,
                               header)
    nib.save(data_nii, saving_path + '/test_batch1_underlying_mask_' +
             now_s + '.nii.gz')

    print("\n\n TEST 2: Loading batch normally and checking result.")
    inputs, directions, previous_dirs = batch_sampler.load_batch(batch)
    print("Nb of inputs: {}. Ex of inputs shape: {}, \n"
          "Nb of directions: {}. Ex of direction shape: {}\n"
          "Previous_dirs: {}. Ex of shape (should be x6): {}"
          .format(len(inputs), inputs[0].shape,
                  len(directions), directions[0].shape,
                  len(previous_dirs), previous_dirs[0].shape))


def test_non_lazy(ref, affine, header, saving_path):
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=False)
    fake_dataset.load_data()

    print('\n\n\n=======================Test with batch size 10000 + resample '
          '+ reverse')
    test_batch_loading_no_computations(fake_dataset, 10000, 0.5, ref,
                                       saving_path, reverse_ratio=1)

    print('\n\n\n=======================Test with batch size 10000 + resample '
          '+ do cpu computations + axis neighborhood')
    test_batch_loading_computations(fake_dataset, 10000, 0.5, ref,
                                    affine, header, saving_path,
                                    neighborhood_type='axes',
                                    neighborhood_radius=[1, 2])

    print('\n\n\n=======================Test with batch size 10000 + resample '
          '+ do cpu computations + grid neighborhood')
    test_batch_loading_computations(fake_dataset, 10000, 0.5, ref,
                                    affine, header, saving_path,
                                    neighborhood_type='grid',
                                    neighborhood_radius=2)


def test_lazy(ref, affine, header, saving_path):
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=True)
    fake_dataset.load_data()

    print('\n\n\n=======================Test with batch size 10000 + resample')
    test_batch_loading_no_computations(fake_dataset, 10000, 0.5, ref,
                                       saving_path)

    print('\n\n\n=======================Test with batch size 10000 + resample '
          '+ computations')
    test_batch_loading_computations(fake_dataset, 10000, 0.5, ref,
                                    affine, header, saving_path)


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

    set_sft_logger_level('WARNING')

    test_non_lazy(args.ref, affine, header, args.saving_path)
    print('\n\n')
    test_lazy(args.ref, affine, header, args.saving_path)
