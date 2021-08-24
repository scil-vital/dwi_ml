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
from dwi_ml.data.dataset.data_lists import (
    DataListForTorch, LazyDataListForTorch)
from dwi_ml.data.dataset.single_subject_containers import (
    SubjectDataAbstract, SubjectData, LazySubjectData)
from dwi_ml.experiment.timer import Timer
from dwi_ml.utils import TqdmLoggingHandler


def find_group_infos(groups: List[str], hdf_subj):
    """
    Separate subject's hdf5 groups intro volume groups or streamline groups
    based on their 'type' attrs.
    """
    volume_groups = []
    streamline_group = None

    for group in groups:
        group_type = hdf_subj[group].attrs['type']
        if group_type == 'volume':
            volume_groups.append(group)
        elif group_type == 'streamlines':
            if streamline_group:
                raise NotImplementedError(
                    "We have not planned yet that you could add two "
                    "groups with type 'streamlines' in the config_file.")
            streamline_group = group
        else:
            raise NotImplementedError(
                "So far, you can only add volume groups in the "
                "groups_config.json. As for the streamlines, they are "
                "added through the option --bundles. Please see the doc "
                "for a json file example. You tried to add data of type: "
                "{}".format(group_type))

    return volume_groups, streamline_group


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
        self.log = None

        # Preparing the dataset
        # type will be dwi_ml.data.dataset.data_lit.DataListForTorch or,
        # in lazy version, LazyDataListForTorch
        self.data_list = None
        self.groups = None
        self.volume_groups = None
        self.streamline_group = None
        self.streamline_id_slice_per_subj = None
        self.total_streamlines = None
        self.streamline_lengths_mm = None

    @property
    def attributes(self) -> Dict[str, Any]:
        all_params = {
            'hdf5_path': self.hdf5_path,
            'name': self.name,
            'set': self.set,
            'taskman_managed': self.taskman_managed,
        }
        return all_params

    def load_data(self):
        """
        Load raw dataset into memory.
        """
        logging.debug("Now opening file {}".format(self.hdf5_path))
        with h5py.File(self.hdf5_path, 'r') as hdf_file:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            database_version = hdf_file.attrs["version"]
            logging.info("Reading hdf file: {}".format(self.hdf5_path))
            logging.info("Database version: {}".format(database_version))
            subject_keys = sorted(hdf_file.attrs[self.set])
            logging.debug('hdf_file (subject) keys for the {} set '
                          'are: {}'.format(self.set, subject_keys))
            self.groups = list(hdf_file[subject_keys[0]].keys())
            logging.debug('Groups of data are: {}'.format(self.groups))

            # Build empty data_list (lazy or not) and initialize values
            self.data_list = self._build_data_list(hdf_file)
            self.streamline_id_slice_per_subj = defaultdict(slice)
            self.total_streamlines = 0
            streamline_lengths_mm_list = []

            # Prepare log to work with tqdm. Use self.log instead of logging
            # inside any tqdm loop
            self.log = logging.getLogger('for_tqdm' + str(datetime.now()))
            self.log.setLevel(logging.root.level)
            self.log.addHandler(TqdmLoggingHandler())
            self.log.propagate = False

            # Using tqdm progress bar, load all subjects from hdf_file
            for subject_id in tqdm.tqdm(subject_keys, ncols=100,
                                        disable=self.taskman_managed):
                if self.volume_groups is None:
                    self.volume_groups, self.streamline_group = \
                        find_group_infos(self.groups, hdf_file[subject_id])

                # Create subject's container
                # (subject = each group in groups_config + streamline)
                # Uses SubjectData or LazySubjectData based on the class
                # calling this method. In the lazy case, the hdf_file is not
                # passed so subject information will basically be empty.
                self.log.debug('* Creating subject "{}": '.format(subject_id))
                subj_data = self._init_subj_from_hdf(
                    hdf_file=hdf_file, subject_id=subject_id,
                    volume_groups=self.volume_groups,
                    streamline_group=self.streamline_group, log=self.log)

                # Add subject to the list
                subj_idx = self.data_list.add_subject(subj_data, self.log)

                # Arrange streamlines
                # In the lazy case, we need to load the data first using the
                # __getitem__ from the datalist, and passing the hdf handle.
                # For the non-lazy version, this does nothing.
                subj_data = subj_data.with_handle(hdf_file)
                streamline_lengths_mm_list = self._arrange_streamlines(
                    subj_data, subj_idx, streamline_lengths_mm_list)

                self.log.debug("* Done")

        # Arrange final data properties
        self.streamline_lengths_mm = np.concatenate(streamline_lengths_mm_list,
                                                    axis=0)

    def _arrange_streamlines(self,
                             subj_data: Union[SubjectData, LazySubjectData],
                             subj_idx: int, streamline_lengths_mm_list: List):
        """
        Concatenating streamlines but remembering which id was which subject's
        streamline.

        The subj_data.streamlines depend on the class: np.Array in the non-lazy
        version, or property in the lazy version, returning streamlines only if
        a handle is present.
        """
        self.log.debug("*    Arranging streamlines per ID.")

        if type(subj_data) == LazySubjectData:
            if subj_data.hdf_handle is None:
                self.log.debug("*    Subject's handle must be present!")

        # Assign a unique ID to every streamline
        n_streamlines = len(subj_data.sft_data.streamlines)
        self.log.debug("*    Subject had {} streamlines."
                       .format(n_streamlines))

        self.streamline_id_slice_per_subj[subj_idx] = slice(
            self.total_streamlines, self.total_streamlines + n_streamlines)

        # Update total nb of streamlines in the dataset
        self.total_streamlines += n_streamlines

        # Get number of timesteps per streamline
        streamline_lengths_mm_list.append(
            np.array(subj_data.sft_data.streamlines.lengths_mm,
                     dtype=np.int16))

        return streamline_lengths_mm_list

    @staticmethod
    def _build_data_list(hdf_file):
        raise NotImplementedError

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups,
                            streamline_group, log):
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

    def __len__(self):
        return self.total_streamlines

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        """
        - If idx: int, it is a unique item database streamline identifier.
            Then, returns a (streamline_id, subject_id) associated with it.
        - If idx: tuple, it is already a (streamline_id, subj_id). Then this
            function does nothing but necessary. Will be used by torch.
        """
        if type(idx) is tuple:
            return idx
        else:
            for subjid, y_slice in self.streamline_id_slice_per_subj.items():
                if y_slice.start <= idx < y_slice.end:
                    streamlineid = idx - y_slice.start
                    return streamlineid, subjid
            raise ValueError("Could not find (streamlineid, subjid) "
                             "from idx: {}".format(idx))


class MultiSubjectDataset(MultiSubjectDatasetAbstract):
    """Dataset containing multiple SubjectData objects saved in a DataList."""
    def __init__(self, hdf5_path: str, subjs_set: str, name: str = None,
                 taskman_managed: bool = False):
        super().__init__(hdf5_path, subjs_set, name, taskman_managed)

        # This will accelerate verification of laziness, compared to
        # is_instance(data, LazyMultiSubjectDataset)
        self.is_lazy = False

    @property
    def attributes(self):
        all_params = super().attributes
        other_params = {
            'is_lazy': self.is_lazy
        }
        all_params.update(other_params)
        return all_params

    @staticmethod
    def _build_data_list(hdf_file):
        """ hdf_file not used. But this is used by load, and the lazy version
        uses this parameter. Keeping the same signature."""
        return DataListForTorch()

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups,
                            streamline_group, log):
        """Non-lazy version: using the class's method init_from_hdf,
        which already loads everyting. Encapsulating this call in a method
        because the lazy version will skip the hdf_file to init subject."""
        return SubjectData.init_from_hdf(subject_id, volume_groups,
                                         streamline_group, hdf_file, log)

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
            'is_lazy': self.is_lazy
        }
        all_params.update(other_params)
        return all_params

    @staticmethod
    def _build_data_list(hdf_file):
        assert hdf_file is not None

        return LazyDataListForTorch(hdf_file)

    @staticmethod
    def _init_subj_from_hdf(hdf_file, subject_id, volume_groups,
                            streamline_group, log):
        """Lazy version: not using hdf_file arg, so no hdf_handle is saved yet
        and data is not loaded."""

        hdf_file = None

        return LazySubjectData.init_from_hdf(subject_id, volume_groups,
                                             streamline_group, hdf_file, log)

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


def init_dataset(is_lazy: bool, hdf5_filename: str, subjs_set: str,
                 name: str = None, taskman_managed: bool = None,
                 cache_size: int = None, **unused_kwargs):
    if is_lazy:
        dataset = LazyMultiSubjectDataset(hdf5_filename, subjs_set, name,
                                          taskman_managed, cache_size)
    else:
        dataset = MultiSubjectDataset(hdf5_filename, subjs_set, name,
                                      taskman_managed)

    with Timer("Loading dataset for {}".format(subjs_set), newline=True,
               color='blue'):
        dataset.load_data()

    logging.debug("Unused kwargs are: {}".format(unused_kwargs))

    return dataset
