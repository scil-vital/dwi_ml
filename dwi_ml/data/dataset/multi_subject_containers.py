# -*- coding: utf-8 -*-
"""
This multisubject dataset:
    - Reads data in the given hdf5 dataset. Keys are the subjects.
    - For each subject, keys are either:
        - Groups to load as mri volume, with attrs 'type' = 'volume' and
        'affine' = the affine.
        - Streamlines to load, with attrs 'type' = 'streamlines' and with
         datasets 'streamlines/data', 'streamlines/offsets',
         'streamlines/lengths', 'streamlines/euclidean_lengths'.

You can use this multisubject dataset as base for your batch sampler. Create
it to get the data from chosen groups based on your model. See for instance
dwi_ml.model.batch_sampler.TrainingBatchSamplerOneInputVolume.
"""
from collections import defaultdict
from datetime import datetime
import logging
import os
from typing import List, Tuple, Union, Dict, Any

import h5py
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
import tqdm

from dwi_ml.cache.cache_manager import SingleThreadCacheManager
from dwi_ml.data.dataset.subjects_list_containers import (
    SubjectsDataList, LazySubjectsDataList)
from dwi_ml.data.dataset.single_subject_containers import (
    SubjectDataAbstract, SubjectData, LazySubjectData)
from dwi_ml.data.dataset.utils import find_groups_info
from dwi_ml.utils import TqdmLoggingHandler


class MultiSubjectDatasetAbstract(Dataset):
    """Dataset containing multiple SubjectData or LazySubjectData objects,
    saved in a DataList or LazyDataList. Abstract class to be implemented below
    for cases lazy or non-lazy.

    Based on torch's dataset class. Provides functions for a DataLoader to
    iterate over data and process batches.
    """
    def __init__(self, hdf5_path: str, subjs_set: str, name: str = None,
                 taskman_managed: bool = False):
        """
        Parameters
        ----------
        hdf5_path: str
            Path to the hdf5 file containing the data.
        subjs_set: str
            Either 'training_subjs' or 'validation_subjs'. The subjects to
            load when using self.load_data(). This refers to the attrs in the
            hdf5 file containing the list of subjects for the training dataset
            and the validation dataset.
        name: str
           Name of the dataset, optional.
        taskman_managed: bool
            Enable or disable que tqdm progress bar.
        """
        # Dataset info
        self.hdf5_path = hdf5_path
        self.name = name if name else os.path.basename(self.hdf5_path)
        self.set = subjs_set
        if not (subjs_set == 'training_subjs' or
                subjs_set == 'validation_subjs'):
            raise ValueError("The MultisubjectDataset set should be either "
                             "'training_subjs' or 'validation_subjs' but we "
                             "received {}".format(subjs_set))

        # Concerning the memory usage:
        self.taskman_managed = taskman_managed

        # Prepare log to work with tqdm. Use self.log instead of logging
        # inside any tqdm loop
        self.log = logging.getLogger('for_tqdm' + str(datetime.now()))
        self.log.setLevel(logging.root.level)
        self.log.addHandler(TqdmLoggingHandler())
        self.log.propagate = False

        # Preparing the dataset
        # type will be dwi_ml.data.dataset.data_list.DataListForTorch or,
        # in lazy version, LazyDataListForTorch
        self.data_list = None
        self.volume_groups = [str]
        self.streamline_groups = [str]
        self.nb_features = [int]
        self.streamline_ids_per_subj = [slice]
        self.total_streamlines = [int]
        self.streamline_lengths_mm = [list]

        self.is_lazy = None

    @property
    def attributes(self) -> Dict[str, Any]:
        all_params = {
            'hdf5_path': self.hdf5_path,
            'name': self.name,
            'set': self.set,
            'taskman_managed': self.taskman_managed,
            'volume_groups': self.volume_groups,
            'nb_features': self.nb_features,
            'streamline_groups': self.streamline_groups,
            'is_lazy': self.is_lazy
        }
        return all_params

    def load_data(self):
        """
        Load raw dataset into memory.
        """
        with h5py.File(self.hdf5_path, 'r') as hdf_file:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            if hdf_file.attrs["version"] < 2:
                logging.warning("Current dwi_ml version should work with "
                                "hdf database version >= 2. This could fail."
                                "database version: {}"
                                .format(hdf_file.attrs["version"]))

            # Basing group names on the first subject
            logging.debug("Loading the first subject's group information. "
                          "Others should fit")
            subject_keys = sorted(hdf_file.attrs[self.set])
            self.volume_groups, self.nb_features, self.streamline_groups = \
                find_groups_info(hdf_file, subject_keys[0], self.log)

            logging.debug('hdf_file (subject) keys for the {} set '
                          'are: {}'.format(self.set, subject_keys))

            # Build empty data_list (lazy or not) and initialize values
            self.data_list = self._build_data_list(hdf_file)
            self.streamline_ids_per_subj = \
                [defaultdict(slice) for _ in self.streamline_groups]
            self.total_streamlines = [0 for _ in self.streamline_groups]
            streamline_lengths_mm_list = [[] for _ in self.streamline_groups]

            # Using tqdm progress bar, load all subjects from hdf_file
            for subject_id in tqdm.tqdm(subject_keys, ncols=100,
                                        disable=self.taskman_managed):
                # Create subject's container
                # Uses SubjectData or LazySubjectData based on the class
                # calling this method. In the lazy case, the hdf_file is not
                # passed so subject information will basically be empty.
                self.log.debug('* Creating subject "{}": '.format(subject_id))
                subj_data = self._init_subj_from_hdf(
                    hdf_file, subject_id, self.volume_groups, self.nb_features,
                    self.streamline_groups, self.log)

                # Add subject to the list
                subj_idx = self.data_list.add_subject(subj_data, self.log)

                # Arrange streamlines
                # In the lazy case, we need to load the data first using the
                # __getitem__ from the datalist, and passing the hdf handle.
                # For the non-lazy version, this does nothing.
                subj_data = subj_data.with_handle(hdf_file)
                for i in range(len(self.streamline_groups)):
                    streamline_lengths_mm_list[i] = self._arrange_streamlines(
                        subj_data, subj_idx, streamline_lengths_mm_list[i], i)

                self.log.debug("* Done")

        # Arrange final data properties
        self.streamline_lengths_mm = \
            [np.concatenate(streamline_lengths_mm_list[i], axis=0)
             for i in range(len(self.streamline_groups))]

    def _arrange_streamlines(
            self, subj_data: Union[SubjectData, LazySubjectData],
            subj_idx: int, group_streamline_lengths_mm_list: List,
            group_idx: int):
        """
        Concatenating streamlines but remembering which id was which subject's
        streamline.

        The subj_data.streamlines depend on the class: np.Array in the non-lazy
        version, or property in the lazy version, returning streamlines only if
        a handle is present.
        """
        group = self.streamline_groups[group_idx]

        self.log.debug("     => Arranging streamlines per ID for group '{}'."
                       .format(group))

        # Assign a unique ID to every streamline
        n_streamlines = len(subj_data.sft_data_list[group_idx].streamlines)
        self.log.debug("        Subject had {} streamlines for this group."
                       .format(n_streamlines))

        # Assigning these id in the dict for this group
        self.streamline_ids_per_subj[group_idx][subj_idx] = \
            slice(self.total_streamlines[group_idx],
                  self.total_streamlines[group_idx] + n_streamlines)

        # Update total nb of streamlines in the dataset for this group
        self.total_streamlines[group_idx] += n_streamlines

        # Get number of timesteps per streamline for this group
        group_streamline_lengths_mm_list.append(
            np.array(subj_data.sft_data_list[group_idx].streamlines.lengths_mm,
                     dtype=np.int16))

        return group_streamline_lengths_mm_list

    @staticmethod
    def _build_data_list(hdf_file):
        raise NotImplementedError

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups, nb_features,
                            streamline_groups, log):
        raise NotImplementedError

    """
    The following methods will help the batch sampler to access specific data
    """
    def get_subject_data(self, subj_idx: int) -> SubjectDataAbstract:
        raise NotImplementedError

    def get_subject_mri_group_as_tensor(self, subj_idx: int, group_idx: int,
                                        device: torch.device,
                                        non_blocking: bool = False):
        """Note. Device is not important for non-lazy version but keeping the
        same signature"""
        raise NotImplementedError

    def __getitem__(self, idx: Tuple[int, int]):
        """
        See here for more information on how the dataloader can use the
        dataset: https://pytorch.org/docs/stable/data.html
        Here, we create a 'map-style' dataset, meaning the the dataloader will
        use dataset[idx]. We could be loading the data here but we will
        actually do it after, in the collate_fn=load_batch, to make sure we
        load one once the data per subject. Thus, this function does nothing
        but passes idx. The dataloader will iterate and pass a list of idx
        (thus a list of tuples) to load_batch.

        Idx: tuple[int, int]
            It is already a (streamline_id, subj_id).
        """
        if not isinstance(idx, tuple):
            raise ValueError("Idx ({}) was not a tuple. Please verify how "
                             "your batch sampler created the list of idx to "
                             "fetch.".format(idx))
        return idx


class MultiSubjectDataset(MultiSubjectDatasetAbstract):
    """Dataset containing multiple SubjectData objects saved in a DataList."""
    def __init__(self, hdf5_path: str, subjs_set: str, name: str = None,
                 taskman_managed: bool = False):
        super().__init__(hdf5_path, subjs_set, name, taskman_managed)

        # This will accelerate verification of laziness, compared to
        # is_instance(data, LazyMultiSubjectDataset)
        self.is_lazy = False

    @staticmethod
    def _build_data_list(hdf_file):
        """ hdf_file not used. But this is used by load, and the lazy version
        uses this parameter. Keeping the same signature."""
        return SubjectsDataList()

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups, nb_features,
                            streamline_groups, log):
        return SubjectData.init_from_hdf(
            subject_id, log, hdf_file,
            (volume_groups, nb_features, streamline_groups))

    def get_subject_data(self, subj_idx: int) -> SubjectData:
        """Here, data_list is a DataListForTorch, and its elements are
        SubjectData."""
        return self.data_list[subj_idx]

    def get_subject_mri_group_as_tensor(self, subj_idx: int, group_idx: int,
                                        device: torch.device = None,
                                        non_blocking: bool = False):
        """Different in lazy version. Here, get_subject_data is a SubjectData,
         its mri_data is a List[SubjectMRIData], all already loaded.
         mri_group_idx corresponds to the group number from the config_file."""
        mri = self.get_subject_data(subj_idx).mri_data_list[group_idx]
        volume_data = mri.as_tensor
        volume_data.to(device=device, non_blocking=non_blocking)

        return volume_data


class LazyMultiSubjectDataset(MultiSubjectDatasetAbstract):
    """Dataset containing multiple LazySubjectData objects saved in a
    LazyDataList."""
    def __init__(self, hdf5_path: str, subjs_set: str, name: str = None,
                 taskman_managed: bool = False, cache_size: int = 0):
        super().__init__(hdf5_path, subjs_set, name, taskman_managed)

        # In case `None` was passed explicitly, change cache_size:
        self.cache_size = cache_size or 0

        # This is important. HDF5 file opening should be done in the
        # __get_item__ method or similar, AFTER the worker sub-processes
        # have started. This way, all sub-processes will have their own
        # HDF handle. Otherwise, there may be data corruption because
        # h5py is not thread-safe.
        self.hdf_handle = None
        if self.cache_size > 0:
            self.volume_cache_manager = None

        # This will accelerate verification of laziness, compared to
        # is_instance(data, LazyMultiSubjectDataset)
        self.is_lazy = True

    @property
    def attributes(self):
        all_params = super().attributes
        other_params = {
            'cache_size': self.cache_size,
        }
        all_params.update(other_params)
        return all_params

    @staticmethod
    def _build_data_list(hdf_file):
        assert hdf_file is not None

        return LazySubjectsDataList(hdf_file)

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups, nb_features,
                            streamline_groups, log):
        return LazySubjectData.init_from_hdf(
            subject_id, log, hdf_file,
            (volume_groups, nb_features, streamline_groups))

    def get_subject_data(self, item) -> LazySubjectData:
        """Contains both MRI data and streamlines. Here, data_list is a
        LazyDataListForTorch, and its elements are LazySubjectData, which
        means we need to open a handle to read the hdf5 file and read the data.
        """
        if self.hdf_handle is None:
            self.hdf_handle = h5py.File(self.hdf5_path, 'r')
        return self.data_list[(item, self.hdf_handle)]

    def get_subject_mri_group_as_tensor(self, subj_idx: int, group_idx: int,
                                        device: torch.device = None,
                                        non_blocking: bool = False):
        """Here, get_subject_data is a LazySubjectData, its mri_data is a
        List[LazySubjectMRIData], not loaded yet but we will load it now using
        as_tensor.
        mri_group_idx corresponds to the group number from the config_file."""
        if self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                volume_data = self.volume_cache_manager[subj_idx]
            except KeyError:
                # volume_data isn't cached; fetch and cache it
                # This will open a hdf handle if it not created yet.
                mri = self.get_subject_data(subj_idx).mri_data_list[group_idx]
                volume_data = mri.as_tensor

                # Send volume_data on device and keep it there while it's
                # cached
                volume_data = volume_data.to(device=device,
                                             non_blocking=non_blocking)

                self.volume_cache_manager[subj_idx] = volume_data
            return volume_data
        else:
            # No cache is used
            mri = self.get_subject_data(subj_idx).mri_data_list[group_idx]
            volume_data = mri.as_tensor
            volume_data.to(device=device, non_blocking=non_blocking)
            return volume_data

    def __del__(self):
        if self.hdf_handle is not None:
            self.hdf_handle.close()


def init_dataset(lazy: bool, hdf5_filename: str, subjs_set: str,
                 name: str = None, taskman_managed: bool = None,
                 cache_size: int = None, **_):
    if lazy:
        dataset = LazyMultiSubjectDataset(hdf5_filename, subjs_set, name,
                                          taskman_managed, cache_size)
    else:
        dataset = MultiSubjectDataset(hdf5_filename, subjs_set, name,
                                      taskman_managed)

    dataset.load_data()

    return dataset
