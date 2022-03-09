#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
import os
import tempfile

import h5py
import numpy as np
import torch
from scilpy.io.fetcher import fetch_data, get_home, get_testing_files_dict

from dwi_ml.data.dataset.multi_subject_containers import \
    MultiSubjectDataset, MultisubjectSubset
from dwi_ml.data.dataset.mri_data_containers import MRIData, LazyMRIData
from dwi_ml.data.dataset.single_subject_containers import \
    SubjectData, LazySubjectData
from dwi_ml.data.dataset.subjectdata_list_containers import \
    SubjectsDataList, LazySubjectsDataList
from dwi_ml.data.dataset.streamline_containers import \
    SFTData, LazySFTData, _LazyStreamlinesGetter


#fetch_data(get_testing_files_dict(), keys=['dwiml.zip'])
#tmp_dir = tempfile.TemporaryDirectory()

expected_volume_groups = ['input', 'wm_mask']
expected_streamline_groups = ['streamlines']
expected_nb_features = [2, 1]  # input = t1 + fa. wm_mask = wm.
expected_nb_streamlines = [3827]  # from the fornix file
expected_nb_subjects = 1
expected_first_subj_name = 'subjX'
expected_mri_shape = [138, 168, 134]


def test_multisubjectdataset():

    logging.basicConfig(level='DEBUG')

    logging.debug("\n"
                  "Unit test: previous dirs\n"
                  "------------------------")

    #os.chdir(os.path.expanduser(tmp_dir.name))
    #hdf5_filename = os.path.join(get_home(), 'dwiml', 'hdf5_file.hdf5')
    home = os.path.expanduser("~")
    hdf5_filename = os.path.join(
        home, 'Bureau/data_for_tests_dwi_ml/hdf5_file.hdf5')

    _non_lazy_version(hdf5_filename)
    logging.debug('\n\n')
    _lazy_version(hdf5_filename)


def _verify_multisubject_dataset(dataset):
    logging.debug("\n====== Testing properties of the main "
                  "MultiSubjectDataset")
    assert isinstance(dataset, MultiSubjectDataset)
    assert dataset.volume_groups == expected_volume_groups
    assert dataset.streamline_groups == expected_streamline_groups
    assert dataset.nb_features == expected_nb_features


def _verify_training_set(training_set):
    logging.debug("\n====== Testing properties of the training set:\n")

    assert isinstance(training_set, MultisubjectSubset)
    assert training_set.nb_subjects == expected_nb_subjects
    assert training_set.volume_groups == expected_volume_groups
    assert training_set.streamline_groups == expected_streamline_groups
    assert training_set.nb_features == expected_nb_features
    assert training_set.total_nb_streamlines == expected_nb_streamlines
    # logging.debug("  - Lengths: {}".format(training_set.streamline_lengths))
    # logging.debug("  - Lengths_mm: {}"
    #               .format(training_set.streamline_lengths_mm))
    # logging.debug("  - Streamline ids per subj: {}"
    #               .format(training_set.streamline_ids_per_subj))


def _verify_data_list(subj_data_list):
    assert len(subj_data_list) == expected_nb_subjects


def _verify_subj_data(subj):
    assert subj.subject_id == expected_first_subj_name
    assert subj.volume_groups == expected_volume_groups
    assert subj.streamline_groups == expected_streamline_groups
    assert subj.nb_features == expected_nb_features


def _verify_mri(mri_data, training_set):
    # Non-lazy: getting data as tensor directly.
    # Lazy: it should load it.
    assert isinstance(mri_data.as_tensor, torch.Tensor)
    assert list(mri_data.as_tensor.shape[0:3]) == expected_mri_shape

    # This should also get it.
    volume0 = training_set.get_volume_verify_cache(0, 0)
    assert isinstance(volume0, torch.Tensor)


def _verify_sft_data(sft_data):
    assert len(sft_data.streamlines) == expected_nb_streamlines[0]
    assert len(sft_data.streamlines[0][0]) == 3  # a x, y, z coordinate
    assert len(sft_data.as_sft().streamlines) == expected_nb_streamlines[0]


def _non_lazy_version(hdf5_filename):
    logging.debug("\n\n**========= NON-LAZY =========\n\n")
    dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=False,
                                  lazy=False)
    dataset.load_data()
    _verify_multisubject_dataset(dataset)

    training_set = dataset.training_set
    assert training_set.cache_size == 0
    _verify_training_set(training_set)

    logging.debug("\n====== Testing properties of the SubjectDataList:\n")
    subj_data_list = training_set.subjs_data_list
    assert isinstance(subj_data_list, SubjectsDataList)
    _verify_data_list(subj_data_list)

    logging.debug("\n====== Testing properties of a SingleSubjectDataset:\n")
    subj0 = training_set.subjs_data_list[0]
    assert isinstance(subj0, SubjectData)
    _verify_subj_data(subj0)

    logging.debug("\n====== Testing properties of his first MRIData:\n")
    mri_data = subj0.mri_data_list[0]
    assert isinstance(mri_data, MRIData)
    # Non-lazy: the _data should already be loaded.
    assert isinstance(mri_data._data, np.ndarray)
    _verify_mri(mri_data, training_set)

    logging.debug("\n====== Testing properties of his first SFTData:\n")
    sft_data = subj0.sft_data_list[0]
    assert isinstance(sft_data, SFTData)
    _verify_sft_data(sft_data)


def _lazy_version(hdf5_filename):
    logging.debug("\n\n**========= LAZY =========\n\n")
    dataset = MultiSubjectDataset(hdf5_filename, taskman_managed=True,
                                  lazy=True, subset_cache_size=1)
    dataset.load_data()
    _verify_multisubject_dataset(dataset)

    training_set = dataset.training_set
    assert training_set.cache_size == 1
    _verify_training_set(training_set)

    print("\n====== Testing properties of the LAZY SubjectDataList:\n")
    subj_data_list = training_set.subjs_data_list
    assert isinstance(subj_data_list, LazySubjectsDataList)
    _verify_data_list(subj_data_list)

    logging.debug("\n====== Testing properties of a LAZY "
                  "SingleSubjectDataset:\n")

    # Directly accessing (_getitem_) should bug: need to send a hdf_handle.
    failed = False
    try:
        _ = training_set.subjs_data_list[0]
    except AssertionError:
        failed = True
    assert failed

    # Accessing through open_handle_and_getitem
    subj0 = training_set.subjs_data_list.open_handle_and_getitem(0)
    assert isinstance(subj0, LazySubjectData)
    _verify_subj_data(subj0)

    logging.debug("\n====== Testing properties of his first LAZY MRIData:\n")
    mri_data = subj0.mri_data_list[0]
    assert isinstance(mri_data, LazyMRIData)
    # Lazy: data should be a hdf5 group)
    assert isinstance(mri_data._data, h5py.Dataset)
    _verify_mri(mri_data, training_set)

    logging.debug("\n====== Testing properties of his first Lazy SFTData:\n")
    sft_data = subj0.sft_data_list[0]
    assert isinstance(sft_data, LazySFTData)
    assert isinstance(sft_data.streamlines, _LazyStreamlinesGetter)
    _verify_sft_data(sft_data)
