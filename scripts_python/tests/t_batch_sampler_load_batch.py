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
from dwi_ml.training.batch_samplers import BatchStreamlinesSamplerOneInput


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
    p.add_argument('input_group_name',
                   help='Input volume group name.')
    p.add_argument('ref',
                   help='Ref MRI volume to save the streamlines')
    p.add_argument('saving_path',
                   help='Path for saving tests (.trk, .nii)')
    return p.parse_args()


def t_batch_loading_no_computations(
        fake_dataset, ref, saving_path, input_group_name,
        noise_size: float = 0, noise_variability: float = 0,
        reverse_ratio: float = 0, split_ratio: float = 0):
    now = datetime.now().time()

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second
    batch_sampler = BatchStreamlinesSamplerOneInput(
        training_set, 'streamlines', max_chunk_size=256,
        max_batch_size=10000, rng=rng_seed,
        nb_subjects_per_batch=1, cycles=1, step_size=0.75,
        compress=False, split_ratio=split_ratio,
        noise_gaussian_size=noise_size,
        noise_gaussian_variability=noise_variability,
        reverse_ratio=reverse_ratio, input_group_name=input_group_name,
        wait_for_gpu=True, neighborhood_type='axes', neighborhood_radius=1)

    print('Iterating once on sampler...')
    batch_generator = batch_sampler.__iter__()
    # Loop on batches but get the first one
    batch = []
    for batch in batch_generator:
        break

    logging.root.setLevel('DEBUG')

    print('\nLoading batch (wait for gpu = true, should not load inputs!)')
    batch_streamlines, ids = batch_sampler.load_batch(batch)

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


def t_batch_loading_computations(
        fake_dataset, ref, affine, header, saving_path, input_group_name,
        neighb_type, neighb_radius):
    set_sft_logger_level('WARNING')
    now = datetime.now().time()

    training_set = fake_dataset.training_set

    # Initialize batch sampler
    print('Initializing sampler...')
    rng_seed = now.minute * 100 + now.second
    batch_sampler = BatchStreamlinesSamplerOneInput(
        training_set, 'streamlines', max_chunk_size=256,
        max_batch_size=10000, rng=rng_seed,
        nb_subjects_per_batch=1, cycles=1, step_size=0.5, compress=False,
        noise_gaussian_size=0,
        noise_gaussian_variability=0, split_ratio=0, reverse_ratio=0,
        input_group_name=input_group_name,
        wait_for_gpu=False, neighborhood_type=neighb_type,
        neighborhood_radius=neighb_radius)

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
    batch_streamlines, ids, inputs_tuple = batch_sampler.load_batch(
        batch, save_batch_input_mask=True)
    batch_input_masks, batch_inputs = inputs_tuple

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
    batch_streamlines, ids, inputs = batch_sampler.load_batch(batch)
    print("Nb of inputs: {}. Ex of inputs shape: {}"
          .format(len(inputs), inputs[0].shape))
    logging.root.setLevel('INFO')


def t_non_lazy(args, affine, header):
    print("\n\n========= NON-LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    dataset = MultiSubjectDataset(args.hdf5_filename, lazy=False,
                                  experiment_name='test',
                                  taskman_managed=True, cache_size=None)
    dataset.load_data()

    print('\n\n===================== A. Test with reverse + split')
    t_batch_loading_no_computations(
        dataset, args.ref, args.saving_path, args.input_group_name,
        reverse_ratio=0.5, split_ratio=0.5)

    print('\n\n==================== B. Test with batch size 10000 + resample '
          '+ do cpu computations + axis neighborhood')
    t_batch_loading_computations(
        dataset, args.ref, affine, header, args.saving_path,
        args.input_group_name, 'axes', [2, 4])

    print('\n\n\n=================== C. Test with batch size 10000 + resample '
          '+ do cpu computations + grid neighborhood')
    t_batch_loading_computations(
        dataset, args.ref, affine, header, args.saving_path,
        args.input_group_name, 'grid', 2)


def t_lazy(args, affine, header):
    print("\n\n========= LAZY =========\n\n")

    # Initialize dataset
    print('Initializing dataset...')
    logging.root.setLevel('INFO')
    dataset = MultiSubjectDataset(args.hdf5_filename, lazy=True,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=1)
    dataset.load_data()

    print('\n\n\n=================== A. Test with basic args')
    t_batch_loading_no_computations(
        dataset, args.ref, args.saving_path, args.input_group_name)

    print('\n\n\n==================== B. Test with wait_for_gpu = false')
    t_batch_loading_computations(
        dataset, args.ref, affine, header, args.saving_path,
        args.input_group_name, None, None)


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

    t_non_lazy(args, affine, header)
    print('\n\n')
    t_lazy(args, affine, header)


if __name__ == '__main__':
    main()
