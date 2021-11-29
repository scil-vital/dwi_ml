# -*- coding: utf-8 -*-
"""
This multisubject dataset:
    - Reads data in the given hdf5 dataset.
    - For each subset (training set, validation set):
        - For each subject, keys are either:
            - Groups to load as mri volume, with attrs 'type' = 'volume' and
              'affine' = the affine.
            - Streamlines to load, with attrs 'type' = 'streamlines' and with
              datasets 'streamlines/data', 'streamlines/offsets',
              'streamlines/lengths', 'streamlines/euclidean_lengths'.

You can use the subsets as base for your batch sampler. Create
it to get the data from chosen groups based on your model.
"""
from collections import defaultdict
from datetime import datetime
import logging
from typing import List, Tuple, Union, Dict, Any

import h5py
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
import tqdm

from dwi_ml.cache.cache_manager import SingleThreadCacheManager
from dwi_ml.data.dataset.utils import find_groups_info
from dwi_ml.data.dataset.subjects_list_containers import (LazySubjectsDataList,
                                                          SubjectsDataList)
from dwi_ml.data.dataset.single_subject_containers import (LazySubjectData,
                                                           SubjectData)
from dwi_ml.utils import TqdmLoggingHandler


class MultisubjectSubset(Dataset):
    """The MultiSubjectDatasets will contain two subsets: the training set
    and the validation set."""
    def __init__(self, set_name: str, hdf5_file: str, taskman_managed: bool,
                 lazy: bool, log, cache_size: int = 0):

        self.set_name = set_name
        self.hdf5_file = hdf5_file
        self._log = log
        self.taskman_managed = taskman_managed

        self.volume_groups = []  # type: List[str]
        self.nb_features = []  # type: List[int]
        self.streamline_groups = []  # type: List[str]

        # The subjects data list will be either a SubjectsDataList or a
        # LazySubjectsDataList depending MultisubjectDataset.is_lazy.
        self.subjs_data_list = None
        self.nb_subjects = 0

        self.streamline_ids_per_subj = []  # type: List[defaultdict[slice]]
        self.total_nb_streamlines = []  # type: List[int]

        # Remembering heaviness to help the batch sampler.
        # One np array per subj per group.
        # - streamline lengths: in number of timepoints.
        # - lengths_mm: euclidean length (will help to compute the new number
        # of timepoints if we intend to resample the streamlines).
        self.streamline_lengths_mm = []  # type: List[List[int]]
        self.streamline_lengths = []  # type: List[List[int]]

        # If data has been resampled in the hdf5, step_size is set, else None
        self.step_size = None

        self.is_lazy = lazy

        # This is only used in the lazy case, cache_size > 0
        self.cache_size = cache_size
        self.volume_cache_manager = None

    @property
    def params(self) -> Dict[str, Any]:
        all_params = {
            'hdf5_file': self.hdf5_file,
            'taskman_managed': self.taskman_managed,
            'volume_groups': self.volume_groups,
            'nb_features': self.nb_features,
            'streamline_groups': self.streamline_groups,
            'lazy': self.is_lazy,
            'cache_size': self.cache_size,
        }
        return all_params

    def __getitem__(self, idx: Tuple[int, list]):
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
            raise ValueError("Subset: Idx ({}) was not a tuple. Please verify "
                             "how your batch sampler created the list of idx "
                             "to fetch.".format(idx))
        return idx

    def get_volume_verify_cache(self, subj_idx: int, group_idx: int,
                                device: torch.device = torch.device('cpu'),
                                non_blocking: bool = False) -> torch.Tensor:
        """
        There will be one cache manager per subset. This could be moved to
        the MultiSubjectDatasetAbstract but easier to deal with here.

        Loads volume as a tensor. In the case of lazy dataset, checks the cache
        first. If data was not on the cache, loads it and sends it to the
        cache before returning.
        """

        # First verifiy cache (if lazy)
        cache_key = str(subj_idx) + '.' + str(group_idx)
        if self.subjs_data_list.is_lazy and self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                mri_data_tensor = self.volume_cache_manager[cache_key]
                return mri_data_tensor
            except KeyError:
                pass

        # Either non-lazy or if lazy, data was not cached.
        mri_data_tensor = self._get_volume_as_tensor(subj_idx, group_idx,
                                                     device, non_blocking)

        if self.subjs_data_list.is_lazy and self.cache_size > 0:
            # Send volume_data on cache
            self.volume_cache_manager[cache_key] = mri_data_tensor

        return mri_data_tensor

    def _get_volume_as_tensor(self, subj_idx: int, group_idx: int,
                              device: torch.device,
                              non_blocking: bool = False):
        """
        Contrary to get_volume_verify_cache, this does not send data to
        cache for later use.

        Loads volume as a tensor.
        """
        if self.subjs_data_list.is_lazy:
            subj_data = self.subjs_data_list.open_handle_and_getitem(
                subj_idx)
        else:
            subj_data = self.subjs_data_list[subj_idx]

        mri_data_tensor = subj_data.mri_data_list[group_idx].as_tensor
        mri_data_tensor.to(device=device, non_blocking=non_blocking)

        return mri_data_tensor


class MultiSubjectDataset:
    """Dataset containing multiple SubjectData or LazySubjectData objects,
    saved in a DataList or LazyDataList. Abstract class to be implemented below
    for cases lazy or non-lazy.

    Based on torch's dataset class. Provides functions for a DataLoader to
    iterate over data and process batches.
    """
    def __init__(self, hdf5_file: str, taskman_managed: bool, lazy: bool,
                 cache_size: Union[int, None], **_):
        """
        Parameters
        ----------
        hdf5_file: str
            Path to the hdf5 file containing the data.
        taskman_managed: bool
            Enable or disable que tqdm progress bar.
        lazy: bool
            Use lazy or non-lazy data. Lazy data only loads data from the hdf5
            when asked explicitely. Non-lazy loads everything at once in the
            load_data method.
        cache_size: int
            Only useful with lazy data (else, you can set it to None.
            Non-optional to ensure arg is there with lazy data). Size of the
            cache in terms of number of length of the queue (i.e. number of
            volumes).
            NOTE: Real cache size will actually be twice this value as the
            training and validation subsets each have their cache.
        """
        # Dataset info
        self.hdf5_file = hdf5_file

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
        self.volume_groups = [str]
        self.streamline_groups = [str]
        self.nb_features = [int]
        self.step_size = None

        self.is_lazy = lazy
        self.cache_size = cache_size
        if self.is_lazy and self.cache_size is None:
            raise ValueError("For lazy data, the cache size cannot be None. "
                             "Maybe you meant 0?")

        # Preparing the testing set and validation set
        # In non-lazy data, the cache_size is not used.
        self.training_set = MultisubjectSubset(
            'training', hdf5_file, taskman_managed, self.is_lazy, self.log,
            cache_size)
        self.validation_set = MultisubjectSubset(
            'validation', hdf5_file, taskman_managed, self.is_lazy, self.log,
            cache_size)
        self.testing_set = MultisubjectSubset(
            'testing', hdf5_file, taskman_managed, self.is_lazy, self.log,
            cache_size)

    @property
    def params(self) -> Dict[str, Any]:
        all_params = {
            'hdf5_file': self.hdf5_file,
            'taskman_managed': self.taskman_managed,
            'volume_groups': self.volume_groups,
            'nb_features': self.nb_features,
            'streamline_groups': self.streamline_groups,
            'is_lazy': self.is_lazy,
            'cache_size': self.cache_size,
        }
        return all_params

    def load_data(self):
        """
        Load raw dataset into memory.
        """
        with h5py.File(self.hdf5_file, 'r') as hdf_handle:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            if hdf_handle.attrs["version"] < 2:
                logging.warning("Current dwi_ml version should work with "
                                "hdf database version >= 2. This could fail."
                                "database version: {}"
                                .format(hdf_handle.attrs["version"]))

            self.step_size = hdf_handle.attrs['step_size']

            # Basing group names on the first training subject
            logging.debug("Loading the first training subject's group "
                          "information. Others should fit")
            subject_keys = sorted(hdf_handle.attrs['training_subjs'])
            self.volume_groups, self.nb_features, self.streamline_groups = \
                find_groups_info(hdf_handle, subject_keys[0], self.log)

            # Saving the group info in the subsets too
            self.training_set.volume_groups = self.volume_groups
            self.training_set.streamline_groups = self.streamline_groups
            self.training_set.nb_features = self.nb_features
            self.training_set.step_size = self.step_size

            self.validation_set.volume_groups = self.volume_groups
            self.validation_set.streamline_groups = self.streamline_groups
            self.validation_set.nb_features = self.nb_features
            self.validation_set.step_size = self.step_size

            # LOADING
            logging.info("---> LOADING TRAINING SET")
            self._load_subset(self.training_set, hdf_handle)
            logging.info("---> LOADING VALIDATION SET")
            self._load_subset(self.validation_set, hdf_handle)
            logging.info("---> LOADING TESTING SET")
            self._load_subset(self.testing_set, hdf_handle)

    def _load_subset(self, subset: MultisubjectSubset, hdf_handle: h5py.File):
        """
        Load all subjects for this subjset (either training, validation or
        testing).
        """
        # Checking if there are any subjects to load
        subject_keys = sorted(hdf_handle.attrs[subset.set_name + '_subjs'])
        if len(subject_keys) == 0:
            logging.debug("    No subject. Returning empty subset.")
            return
        subset.nb_subjects = len(subject_keys)

        # Build empty data_list (lazy or not) and initialize values
        subset.subjs_data_list = self._build_empty_data_list()
        subset.streamline_ids_per_subj = \
            [defaultdict(slice) for _ in self.streamline_groups]
        subset.total_nb_streamlines = [0 for _ in self.streamline_groups]

        # Remembering heaviness. One np array per subj per group.
        lengths = [[] for _ in self.streamline_groups]
        lengths_mm = [[] for _ in self.streamline_groups]

        # Using tqdm progress bar, load all subjects from hdf_file
        for subj_id in tqdm.tqdm(subject_keys, ncols=100,
                                 disable=self.taskman_managed):
            # Create subject's container
            # Uses SubjectData or LazySubjectData based on the class
            # calling this method.
            self.log.debug("     Creating subject '{}':".format(subj_id))
            subj_data = self._init_subj_from_hdf(
                hdf_handle, subj_id, self.volume_groups, self.nb_features,
                self.streamline_groups, self.log)

            # Add subject to the list
            subj_idx = subset.subjs_data_list.add_subject(subj_data)

            if subj_data.is_lazy:
                self.log.debug("     Temporarily adding hdf handle to "
                               "subj to arrange streamlines.")
                subj_data = subset.subjs_data_list[(subj_idx, hdf_handle)]
                self.log.debug("--> Handle: {}".format(subj_data.hdf_handle))

            # Arrange streamlines
            # In the lazy case, we need to load the data first using the
            # __getitem__ from the datalist, and passing the hdf handle.
            # For the non-lazy version, this does nothing.
            subj_data.add_handle(hdf_handle)
            for i in range(len(self.streamline_groups)):
                subj_sft_data = subj_data.sft_data_list[i]
                n_streamlines = len(subj_sft_data.streamlines)
                self._add_streamlines_ids(subset, n_streamlines, subj_idx, i)
                lengths[i].append(subj_sft_data.lengths)
                lengths_mm[i].append(subj_sft_data.lengths_mm)

        # Arrange final data properties
        subset.streamline_lengths_mm = \
            [np.concatenate(lengths_mm[i], axis=0)
             for i in range(len(self.streamline_groups))]
        subset.streamline_lengths = \
            [np.concatenate(lengths[i], axis=0)
             for i in range(len(self.streamline_groups))]

        # No need to return 'subset': instance attributes are modified
        # in-place.

    @staticmethod
    def _add_streamlines_ids(subset: MultisubjectSubset,
                             n_streamlines: int, subj_idx: int,
                             group_idx: int):
        """
        Concatenating streamlines of a specific subject to group #group_idx
        Remembering which id was which subject's streamline.

        The subj_sft_data.streamlines depend on the class: np.Array in the
        non-lazy version, or property in the lazy version, returning
        streamlines only if a handle is present.
        """
        # Assigning these id in the dict for this group
        start = subset.total_nb_streamlines[group_idx]
        end = subset.total_nb_streamlines[group_idx] + n_streamlines
        subset.streamline_ids_per_subj[group_idx][subj_idx] = slice(start, end)

        # Update total nb of streamlines in the dataset for this group
        subset.total_nb_streamlines[group_idx] += n_streamlines

    def _build_empty_data_list(self):
        if self.is_lazy:
            return LazySubjectsDataList(self.hdf5_file, self.log)
        else:
            return SubjectsDataList(self.hdf5_file, self.log)

    def _init_subj_from_hdf(self, hdf_handle, subject_id, volume_groups,
                            nb_features, streamline_groups, log):
        if self.is_lazy:
            return LazySubjectData.init_from_hdf(
                subject_id, log, hdf_handle,
                (volume_groups, nb_features, streamline_groups))
        else:
            return SubjectData.init_from_hdf(
                subject_id, log, hdf_handle,
                (volume_groups, nb_features, streamline_groups))
