#!/usr/bin/env python
import argparse
from argparse import RawTextHelpFormatter
import logging
from os import path

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset


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


def t_non_lazy(args):
    print("\n\n**========= NON-LAZY =========\n\n")
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=False,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=None)
    fake_dataset.load_data()

    print("\n====== Testing properties of the main MultiSubjectDataset: \n")
    print("  - Type: {}".format(type(fake_dataset)))
    print("  - Step size of the original data: {}"
          .format(fake_dataset.step_size))
    print("  - Volume group, features, streamlines: {}, {}, {}"
          .format(fake_dataset.volume_groups, fake_dataset.nb_features,
                  fake_dataset.streamline_groups))

    print("\n====== Testing properties of the training set:\n")
    training_set = fake_dataset.training_set
    print("  - Type: {}".format(type(training_set)))
    print("  - Nb subjects: {}".format(training_set.nb_subjects))
    print("  - Volume group, features, streamlines: {}, {}, {}"
          .format(training_set.volume_groups, training_set.nb_features,
                  training_set.streamline_groups))
    print("  - Cache size: {}".format(training_set.cache_size))
    print("  - Total nb streamlines: {}"
          .format(training_set.total_nb_streamlines))
    print("  - Lengths: {}".format(training_set.streamline_lengths))
    print("  - Lengths_mm: {}".format(training_set.streamline_lengths_mm))
    print("  - Streamline ids per subj: {}"
          .format(training_set.streamline_ids_per_subj))

    print("\n====== Testing properties of the SubjectDataList:\n")
    subj_data_list = training_set.subjs_data_list
    print("  - type: {}".format(subj_data_list))

    print("\n====== Testing properties of a SingleSubjectDataset:\n")
    subj0 = training_set.subjs_data_list[0]
    print("  - Type: {}".format(type(subj0)))
    print("  - Subj id: {}".format(subj0.subject_id))
    print("  - Volume group, features, streamlines: {}, {}, {}"
          .format(subj0.volume_groups, subj0.nb_features,
                  subj0.streamline_groups))

    print("\n====== Testing properties of the first group's MRIData:\n")
    mri_data = subj0.mri_data_list[0]
    print("  - Type: {}".format(type(subj0)))
    print("  - _data (non-lazy, should be an array): {}, shape {}"
          .format(type(mri_data._data), mri_data._data.shape))
    print("  - as tensor (non-lazy, should be the same): {}"
          .format(type(mri_data.as_tensor)))
    volume0 = training_set.get_volume_verify_cache(0, 0)
    print("  - Getting volume from cache: type: {}".format(type(volume0)))

    print("\n====== Testing properties of the first group's SFTData:\n")
    sft_data = subj0.sft_data_list[0]
    print("  - Type: {}".format(type(sft_data)))
    print("  - Number of streamlines: {} \n"
          "  - First streamline: {}... \n"
          "  - As SFT: {}"
          .format(len(sft_data.streamlines),
                  sft_data.streamlines[0][0],
                  sft_data.as_sft()))


def t_lazy(args):
    print("\n\n**========= LAZY =========\n\n")
    fake_dataset = MultiSubjectDataset(args.hdf5_filename, lazy=True,
                                       experiment_name='test',
                                       taskman_managed=True, cache_size=1)
    fake_dataset.load_data()
    training_set = fake_dataset.training_set

    print("\n====== Testing properties of the LAZY SubjectDataList:\n")
    subj_data_list = training_set.subjs_data_list
    print("  - type: {}".format(subj_data_list))

    print("\n====== Testing properties of a LAZY SingleSubjectDataset:\n")
    print("  - Directly accessing (_getitem_) should bug: need to send "
          "a hdf_handle.")
    try:
        _ = training_set.subjs_data_list[0]
    except AssertionError:
        print("    Try, catch: Yes, bugged.")
    print("  - Accessing through open_handle_and_getitem")
    subj0 = training_set.subjs_data_list.open_handle_and_getitem(0)
    print("  - Subj id: {}".format(subj0.subject_id))
    print("  - Volume group, features, streamlines: {}, {}, {}"
          .format(subj0.volume_groups, subj0.nb_features,
                  subj0.streamline_groups))

    print("\n====== Testing properties of the first group's LAZYMRIData:\n")
    mri_data = subj0.mri_data_list[0]
    print("  - Type: {}".format(type(subj0)))
    print("  - _data (lazy, should be a hdf5 group): {}"
          .format(type(mri_data._data)))
    print("  - as tensor (lazy, should load): {}"
          .format(type(mri_data.as_tensor)))
    volume0 = training_set.get_volume_verify_cache(0, 0)
    print("  - Getting volume from cache: type: {}".format(type(volume0)))

    print("\n====== Testing properties of the first group's LazySFTData:\n")
    sft_data = subj0.sft_data_list[0]
    print("  - Type: {}".format(type(sft_data)))
    streamlines = sft_data.streamlines
    print("  - Streamlines not loaded: {}".format(type(streamlines)))
    sft = sft_data.as_sft()
    print("  - as sft: {}".format(type(sft)))


def main():
    args = parse_args()

    if not path.exists(args.hdf5_filename):
        raise ValueError("The hdf5 file ({}) was not found!"
                         .format(args.hdf5_filename))

    logging.basicConfig(level='DEBUG')

    t_non_lazy(args)
    print('\n\n')
    t_lazy(args)


if __name__ == '__main__':
    main()
