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
from dipy.io.streamline import (save_tractogram, Space, Origin)

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.training.batch_samplers import BatchStreamlinesSamplerOneInputAndPD
from dwi_ml.models.main_models import MainModelAbstractNeighborsPreviousDirs


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
        fake_dataset, ref, saving_path, model,
        noise_size: float = 0, noise_variability: float = 0,
        reverse_ratio: float = 0, split_ratio: float = 0):

    now = datetime.now().time()

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second
    batch_sampler = BatchStreamlinesSamplerOneInputAndPD(
        training_set, 'streamlines', chunk_size=256,
        max_batch_size=10000, rng=rng_seed,
        nb_subjects_per_batch=1, cycles=1, step_size=0.75,
        compress=False, wait_for_gpu=True, normalize_directions=False,
        split_ratio=split_ratio, noise_gaussian_size=noise_size,
        noise_gaussian_variability=noise_variability,
        reverse_ratio=reverse_ratio, model=model)

    print('Iterating once on sampler...')
    batch_generator = batch_sampler.__iter__()
    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        break

    logging.root.setLevel('DEBUG')

    print('\nLoading batch')
    batch_streamlines, _ = batch_sampler.load_batch(batch)

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

    print("Saving subj 0's tractogram {}".format('test_batch1_' + now_s))

    logging.root.setLevel('INFO')

    sft = StatefulTractogram(batch_streamlines, ref, space=Space.VOX,
                             origin=Origin.TRACKVIS)
    save_tractogram(sft, saving_path + '/test_batch_reverse_split_' +
                    now_s + '.trk')

    return batch_streamlines


def test_batch_loading_computations(
        fake_dataset, ref, affine, header, saving_path, model):
    set_sft_logger_level('WARNING')
    now = datetime.now().time()

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second
    batch_sampler = BatchStreamlinesSamplerOneInputAndPD(
        training_set, 'streamlines', chunk_size=256,
        max_batch_size=10000, rng=rng_seed,
        nb_subjects_per_batch=1, cycles=1, step_size=0.5, compress=False,
        wait_for_gpu=False, normalize_directions=True, noise_gaussian_size=0,
        noise_gaussian_variability=0, split_ratio=0, reverse_ratio=0,
        model=model)

    print('Iterating once on sampler...')
    batch_generator = batch_sampler.__iter__()
    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        print('Batch # 1: nb sampled streamlines subj 0 was {}'
              .format(len(batch[0])))
        print("Batch # 1 streamline #1's id: {}".format(batch[0][0]))
        break

    logging.root.setLevel('DEBUG')

    print("\n TEST 1: Debug mode. Saving input coordinates as mask.")
    inputs, directions, previous_dirs = batch_sampler.load_batch(
        batch, save_batch_input_mask=True)

    batch_streamlines, batch_input_masks, batch_inputs = inputs

    print('Nb loaded processed batch streamlines: {}. Streamline 1: {}'
          .format(len(batch_streamlines), batch_streamlines[0][0]))

    millisecond = round(now.microsecond / 10000)
    now_s = str(now.minute * 10000 + now.second * 100 + millisecond)

    print("Saving subj 0's tractogram {}".format('test_batch1_' + now_s))
    sft = StatefulTractogram(batch_streamlines, ref, space=Space.VOX,
                             origin=Origin.TRACKVIS)
    save_tractogram(
        sft, saving_path + '/test_batch_underlying_mask_' + now_s + '.trk')

    print("Saving subj 0's underlying coords mask: {}"
          .format('test_batch1_underlying_mask_' + now_s))
    mask = batch_input_masks[0]
    data_nii = nib.Nifti1Image(np.asarray(mask, dtype=bool), affine,
                               header)
    nib.save(data_nii, saving_path + '/test_batch_underlying_mask_' +
             now_s + '.nii.gz')

    print("\n\n TEST 2: Loading batch normally and checking result.")
    inputs, directions, previous_dirs = batch_sampler.load_batch(batch)
    print("Nb of inputs: {}. Ex of inputs shape: {}, \n"
          "Nb of directions: {}. Ex of direction shape: {}\n"
          "Previous_dirs: {}. Ex of shape (should be x6): {}"
          .format(len(inputs), inputs[0].shape,
                  len(directions), directions[0].shape,
                  len(previous_dirs), previous_dirs[0].shape))
    logging.root.setLevel('INFO')


def test_non_lazy(args, affine, header):
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=False,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=None)
    fake_dataset.load_data()

    print('\n\n===================== A. Test with reverse + split')
    model = create_model(fake_dataset, None, None)
    test_batch_loading_no_computations(
        fake_dataset, args.ref, args.saving_path, model,
        reverse_ratio=0.5, split_ratio=0.5)

    print('\n\n==================== B. Test with batch size 10000 + resample '
          '+ do cpu computations + axis neighborhood')
    model = create_model(fake_dataset, 'axes', [2, 4])
    test_batch_loading_computations(
        fake_dataset, args.ref, affine, header, args.saving_path, model)

    print('\n\n\n=================== C. Test with batch size 10000 + resample '
          '+ do cpu computations + grid neighborhood')
    model = create_model(fake_dataset, 'grid', 2)
    test_batch_loading_computations(
        fake_dataset, args.ref, affine, header, args.saving_path, model)


def test_lazy(args, affine, header):
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=True,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=1)
    fake_dataset.load_data()
    model = create_model(fake_dataset, None, None)

    print('\n\n\n=================== A. Test with basic args')
    test_batch_loading_no_computations(
        fake_dataset, args.ref, args.saving_path, model)

    print('\n\n\n==================== B. Test with wait_for_gpu = false')
    test_batch_loading_computations(
        fake_dataset, args.ref, affine, header, args.saving_path, model)


def create_model(dataset, neighb_type, neighb_radius):
    # print("Nb previous dirs: 3")
    model = MainModelAbstractNeighborsPreviousDirs(
        'test', dataset.nb_features[0], dataset.volume_groups[0], 3,
        neighb_type, neighb_radius)
    return model


def main():
    args = parse_args()
    logging.basicConfig(level='INFO')

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    img = nib.load(args.ref)
    affine = img.affine
    header = img.header

    set_sft_logger_level('WARNING')

    test_non_lazy(args, affine, header)
    print('\n\n')
    test_lazy(args, affine, header)


if __name__ == '__main__':
    main()
