#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from os import path

import numpy as np

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)


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


def test_non_lazy():
    print("**========= NON-LAZY =========")
    fake_dataset = MultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()
    print("**Created a MultiSubjectDataset and loaded training set. "
          "Testing properties : \n\n")

    subj0 = fake_dataset.get_subject_data(0)
    print("**Get_subject_data: \n"
          "    Subject 0, should be SubjectData : {}. \n"
          "    ID: {}. \n"
          "    Volume groups: {}. \n"
          "    Non-lazy mri data should be a list of MRIData: {}. \n"
          "        First volume's _data (is a tensor), shape: {} \n"
          "    Streamline group: '{}' \n"
          "        Number of streamlines: {} \n"
          "        First streamline: {}... \n"
          "        First streamline as sft: {} \n"
          .format(subj0, subj0.subject_id, subj0.volume_groups,
                  subj0.mri_data_list, subj0.mri_data_list[0]._data.shape,
                  subj0.streamline_group,
                  len(subj0.sft_data.streamlines),
                  subj0.sft_data.streamlines[0][0],
                  subj0.sft_data.get_chosen_streamlines_as_sft(0).streamlines[0][0]))

    subj0_volume0_tensor = fake_dataset.get_subject_mri_group_as_tensor(
        0, 0, device=None)
    print("**Get_subject_mri_data_as_tensor from subject 0, volume 0: \n"
          "     Shape {} \n"
          "     First data (nan is normal: outside mask): {} \n"
          "     First volume non-nan mean: {} \n"
          .format(subj0_volume0_tensor.shape, subj0_volume0_tensor[0][0][0],
                  np.nanmean(subj0_volume0_tensor.numpy())))

    del fake_dataset


def test_lazy():
    print("**========= LAZY =========")
    fake_dataset = LazyMultiSubjectDataset(args.hdf5_filename)
    fake_dataset.load_training_data()
    print("**Created a LazyMultiSubjectDataset and loaded training set. "
          "Testing properties : \n\n")

    subj0 = fake_dataset.get_subject_data(0)
    print("**Get_subject_data (in this case, loading from hdf_handle): \n"
          "    Subject 0, should be LazySubjectData : {}. \n"
          "    Handle should be added by now: {}"
          "    ID: {}. \n"
          "    Volume groups: {}. \n"
          "    Lazy mri data should be a list of LazySubjectMRIData: {}. \n"
          "        First volume _data (is not a tensor!): {} \n"
          "    Streamline group: {} \n"
          "        Streamlines (getter not loaded!): {} \n"
          "        First streamline: {} \n"
          "        First streamline as sft: {} \n"
          .format(subj0, subj0.hdf_handle, subj0.subject_id,
                  subj0.volume_groups,
                  subj0.mri_data_list, subj0.mri_data_list[0]._data,
                  subj0.streamline_group, subj0.sft_data.streamlines,
                  subj0.sft_data.streamlines.get_array_sequence()[0][0],
                  subj0.sft_data.get_chosen_streamlines_as_sft(0).streamlines[0][0]))

    subj0_volume0_tensor = fake_dataset.get_subject_mri_group_as_tensor(
        0, 0, device=None)
    print("**Get_subject_mri_data_as_tensor: subject 0, volume 0. \n"
          "     Shape {} \n"
          "     First data (nan is normal: outside mask): {} \n"
          "     First volume non-nan mean: {} \n"
          .format(subj0_volume0_tensor.shape, subj0_volume0_tensor[0][0][0],
                  np.nanmean(subj0_volume0_tensor.numpy())))

    del fake_dataset


if __name__ == '__main__':
    args = parse_args()

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    logging.basicConfig(level='DEBUG')
    rng = np.random.RandomState(seed=1234)

    test_non_lazy()
    print('\n\n')
    test_lazy()
