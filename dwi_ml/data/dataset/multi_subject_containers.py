# -*- coding: utf-8 -*-
import os
from collections import defaultdict
import logging
from typing import List, Tuple, Dict, Any

import h5py
import numpy as np
import torch.multiprocessing
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dwi_ml.cache.cache_manager import SingleThreadCacheManager
from dwi_ml.data.dataset.checks_for_groups import prepare_groups_info
from dwi_ml.data.dataset.mri_data_containers import MRIDataAbstract
from dwi_ml.data.dataset.subjectdata_list_containers import (
    LazySubjectsDataList, SubjectsDataList)
from dwi_ml.data.dataset.single_subject_containers import (LazySubjectData,
                                                           SubjectData)

logger = logging.getLogger('dataset_logger')


class MultisubjectSubset(Dataset):
    """
    Meant to represent the training set, the validation set OR the testing set.

    Contains the list of subjects, the expected list of volumes and streamlines
    for each, and the SingleSubjectDataList, a list of SingleSubjectData, which
    contain the MRI data and streamlines for a given subject.

    Defines a cache, used for lazy data, to avoid reloading the same volume
    many times in a row if some method (ex: batch sampler) is still using that
    subject.

    Based on torch's dataset class. Provides functions for a DataLoader to
    iterate over data and process batches.
    """
    def __init__(self, set_name: str, hdf5_file: str, lazy: bool,
                 cache_size: int = 0):

        self.set_name = set_name
        self.hdf5_file = hdf5_file

        self.volume_groups = []  # type: List[str]
        self.nb_features = []  # type: List[int]
        self.streamline_groups = []  # type: List[str]
        self.contains_connectivity = []  # type: np.ndarray

        # The subjects data list will be either a SubjectsDataList or a
        # LazySubjectsDataList depending on MultisubjectDataset.is_lazy.
        self.subjs_data_list = None
        self.subjects = []  # type:List[str]
        self.nb_subjects = 0

        self.streamline_ids_per_subj = []  # type: List[defaultdict[slice]]

        # One value per streamline group.
        self.total_nb_streamlines = []  # type: List[int]
        self.total_nb_points = []  # type: List[int]

        # Remembering heaviness here to help the batch sampler instead of
        # having to loop on all subjects (again) later.
        # One np.array per subject per group.
        # - lengths: in number of timepoints.
        # - lengths_mm: Euclidean length (will help to compute the new number
        # of timepoints if we intend to resample the streamlines or guess the
        # streamline heaviness with compressed streamlines).
        self.streamline_lengths_mm = []  # type: List[List[int]]
        self.streamline_lengths = []  # type: List[List[int]]

        # Preprocessing information will be found in the hdf5 later.
        self.step_size = None
        self.compress = None

        self.is_lazy = lazy

        # This is only used in the lazy case.
        self.cache_size = cache_size
        self.volume_cache_manager = None

    def close_all_handles(self):
        if self.subjs_data_list.hdf_handle:
            self.subjs_data_list.hdf_handle.close()
            self.subjs_data_list.hdf_handle = None
        for i in range(self.nb_subjects):
            s = self.subjs_data_list[i]
            if s.hdf_handle:
                s.hdf_handle.close()
                s.hdf_handle = None

    def set_subset_info(self, volume_groups, nb_features, streamline_groups,
                        contains_connectivity, step_size, compress):
        self.volume_groups = volume_groups
        self.nb_features = nb_features
        self.streamline_groups = streamline_groups
        self.contains_connectivity = contains_connectivity
        self.step_size = step_size
        self.compress = compress

    @property
    def params(self) -> Dict[str, Any]:
        # Params for init:
        all_params = {
            'set_name': self.set_name,
            'hdf5_file': self.hdf5_file,
            'lazy': self.is_lazy,
            'cache_size': self.cache_size
        }

        # Params that would need to be reset if loaded from a checkpoint:
        all_params.update({
            'volume_groups': self.volume_groups,
            'nb_features': self.nb_features,
            'streamline_groups': self.streamline_groups,
        })
        return all_params

    def __getitem__(self, idx: Tuple[int, list]):
        """
        See here for more information on how the dataloader can use the
        dataset: https://pytorch.org/docs/stable/data.html
        Here, we create a 'map-style' dataset, meaning the the dataloader will
        use dataset[idx]. We could be loading the data here, but we will
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
                                device: torch.device = torch.device('cpu')):
        """
        Get a volume from a specific subject. For lazy data, load it (if not
        already in the cache, or get it from the cache).

        Params
        ------
        subj_idx: int
            Index of the subject to load data from.
        group_idx: int
            Index of the volume to load.
        device:
            Torch device. Used when loading as tensor.

        Returns
        -------
        mri_data: Union[Tensor, DatasetVolume]
        """
        # Note. Developer's choice:
        # There will be one cache manager per subset. This could be moved to
        # the MultiSubjectDatasetAbstract to have only one cache but easier to
        # deal with here.

        # First verify cache
        if device is not None and device.type == 'cuda' and not self.cache_size:
            raise ValueError("Cache size 0 is never recommended with GPU! "
                             "Moving data to GPU everytime!")

        cache_key = str(subj_idx) + '.' + str(group_idx)
        was_cached = False
        if self.cache_size:
            # Initialize the cache if not done
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            # Access the cache
            if cache_key in self.volume_cache_manager:
                mri_data_tensor = self.volume_cache_manager[cache_key]
                was_cached = True

                # User should not change device between calls but just checking
                mri_data_tensor = mri_data_tensor.to(device)

        if not was_cached:
            # Either non-lazy or if lazy, data was not cached.
            # Non-lazy: direct access. Lazy: this loads the whole data.
            logger.debug("Getting a new volume from the dataset.")
            mri_data = self.get_mri_data(subj_idx, group_idx)
            mri_data_tensor = mri_data.get_data_as_tensor(device)

            # Add to cache the tensor (on correct device)
            if self.cache_size:
                logger.debug("PROCESS ID {}. Adding volume to cache"
                             .format(os.getpid()))
                # Send volume_data on cache
                self.volume_cache_manager[cache_key] = mri_data_tensor

        return mri_data_tensor

    def get_mri_data(self, subj_idx: int, group_idx: int,
                     load_it: bool = True) -> MRIDataAbstract:
        """
        Loads volume as a MRIDataAbstract (lazy or non-lazy class instance).

        Contrary to get_volume_verify_cache, this does not send data to
        cache for later use.

        Parameters
        ----------
        subj_idx: int
            The subject id.
        group_idx: int
            The volume group idx.
        load_it: bool
            If data is lazy, get the volume as a LazyMRIData (False) or load it
            as non-lazy (if True).
        """
        if self.subjs_data_list.is_lazy:
            if load_it:
                subj_data = self.subjs_data_list.get_subj_with_handle(subj_idx)
            else:
                subj_data = self.subjs_data_list[subj_idx]
        else:
            subj_data = self.subjs_data_list[subj_idx]

        mri_data = subj_data.mri_data_list[group_idx]

        return mri_data

    def load(self, hdf_handle: h5py.File, subj_id=None):
        """
        Load all subjects for this subjset (either training, validation or
        testing).
        """
        # Checking if there are any subjects to load
        subject_keys = sorted(hdf_handle.attrs[self.set_name + '_subjs'])
        if subj_id is not None:
            if subj_id not in subject_keys:
                raise ValueError(
                    "Given subject ID {} to be loaded in the {} set does not "
                    "exist. \nPossible subjects in that set are: {}"
                    .format(subj_id, self.set_name, subject_keys))
            else:
                subject_keys = [subj_id]
        self.subjects = subject_keys
        self.nb_subjects = len(subject_keys)

        if self.nb_subjects == 0:
            logger.debug("{} set: No subject. Returning empty subset."
                         .format(self.set_name))
            return

        logger.info("LOADING: {} set".format(self.set_name))

        # Build empty data_list (lazy or not) and initialize values
        self.subjs_data_list = self._build_empty_data_list()
        self.streamline_ids_per_subj = \
            [defaultdict(slice) for _ in self.streamline_groups]
        self.total_nb_streamlines = [0 for _ in self.streamline_groups]
        self.total_nb_points = [0 for _ in self.streamline_groups]

        # Remembering heaviness. One np array per subj per group.
        lengths = [[] for _ in self.streamline_groups]
        lengths_mm = [[] for _ in self.streamline_groups]

        ref_group_info = (self.volume_groups, self.nb_features,
                          self.streamline_groups, self.contains_connectivity)

        # Using tqdm progress bar, load all subjects from hdf_file
        with logging_redirect_tqdm(loggers=[logging.root], tqdm_class=tqdm):
            for subj_id in tqdm(subject_keys, ncols=100, total=self.nb_subjects):
                # Create subject's container
                # Uses SubjectData or LazySubjectData based on the class
                # calling this method.
                logger.debug("     Creating subject '{}'.".format(subj_id))
                subj_data = self._init_subj_from_hdf(
                    hdf_handle, subj_id, ref_group_info)

                # Add subject to the list
                subj_idx = self.subjs_data_list.add_subject(subj_data)

                # Arrange streamlines
                # In the lazy case, we need to allow loading the data, passing
                # the hdf handle.
                if subj_data.is_lazy:
                    subj_data.add_handle(hdf_handle)

                for group in range(len(self.streamline_groups)):
                    subj_sft_data = subj_data.sft_data_list[group]
                    n_streamlines = len(subj_sft_data)
                    self._add_streamlines_ids(n_streamlines, subj_idx, group)
                    lengths[group].append(subj_sft_data.lengths)
                    lengths_mm[group].append(subj_sft_data.lengths_mm)

                    # Remove hdf handle
                subj_data.hdf_handle = None

            # Arrange final data properties: Concatenate all subjects
            logging.debug("All subjects added. Final verifications.")
            self.streamline_lengths_mm = \
                [np.concatenate(lengths_mm[group], axis=0)
                 for group in range(len(self.streamline_groups))]
            self.streamline_lengths = \
                [np.concatenate(lengths[i], axis=0)
                 for i in range(len(self.streamline_groups))]
            self.total_nb_points = \
                [sum(self.streamline_lengths[group])
                 for group in range(len(self.streamline_groups))]

    def _add_streamlines_ids(self, n_streamlines: int, subj_idx: int,
                             group_idx: int):
        """
        Concatenating streamlines of a specific subject to group #group_idx
        Remembering which id was which subject's streamline.

        The subj_sft_data.streamlines depend on the class: np.Array in the
        non-lazy version, or property in the lazy version, returning
        streamlines only if a handle is present.
        """
        # Assigning these id in the dict for this group
        start = self.total_nb_streamlines[group_idx]
        end = self.total_nb_streamlines[group_idx] + n_streamlines
        self.streamline_ids_per_subj[group_idx][subj_idx] = slice(start, end)

        # Update total nb of streamlines in the dataset for this group
        self.total_nb_streamlines[group_idx] += n_streamlines

    def _build_empty_data_list(self):
        if self.is_lazy:
            return LazySubjectsDataList(self.hdf5_file, logger)
        else:
            return SubjectsDataList(self.hdf5_file, logger)

    def _init_subj_from_hdf(self, hdf_handle, subject_id, ref_group_info):
        if self.is_lazy:
            return LazySubjectData.init_single_subject_from_hdf(
                subject_id, hdf_handle, ref_group_info)
        else:
            return SubjectData.init_single_subject_from_hdf(
                subject_id, hdf_handle, ref_group_info)


class MultiSubjectDataset:
    """
    This multisubject dataset:
    - Reads data in the given hdf5 dataset.
    - For each subset (training set, validation set):
        - For each subject, keys are either:
            - Groups to load as mri volume, with attrs 'type' = 'volume' and
              'voxres' = the voxel resolution.
            - Streamlines to load, with attrs 'type' = 'streamlines' and with
              datasets 'streamlines/data', 'streamlines/offsets',
              'streamlines/lengths', 'streamlines/euclidean_lengths'.
    """
    def __init__(self, hdf5_file: str, lazy: bool,
                 cache_size: int = 0, log_level=None):
        """
        Params
        ------
        hdf5_file: str
            Path to the hdf5 file containing the data.
        lazy: bool
            Use lazy or non-lazy data. Lazy data only loads data from the hdf5
            when asked explicitely. Non-lazy loads everything at once in the
            load_data method.
        cache_size: int
            Only useful with lazy data. Size of the cache in terms of length of
            the queue (i.e. number of volumes). Default = 0.
            NOTE: Real cache size will actually be twice or trice this value as
            the training, validation and testing sets each have their cache.
        """
        # Dataset info
        self.hdf5_file = hdf5_file

        if log_level is not None:
            logger.setLevel(log_level)

        self.volume_groups = []  # type: List[str]
        self.nb_features = []  # type: List[int]
        self.streamline_groups = []  # type: List[str]
        self.streamlines_contain_connectivity = []

        self.is_lazy = lazy
        self.subset_cache_size = cache_size
        if self.is_lazy and self.subset_cache_size == 0:
            raise ValueError("For lazy data, the cache size cannot be None. "
                             "Maybe you meant 0?")

        # Preparing the testing set and validation set
        # In non-lazy data, the cache_size is not used.
        self.training_set = MultisubjectSubset(
            'training', hdf5_file, self.is_lazy, cache_size)
        self.validation_set = MultisubjectSubset(
            'validation', hdf5_file, self.is_lazy, cache_size)
        self.testing_set = MultisubjectSubset(
            'testing', hdf5_file, self.is_lazy, cache_size)

    @property
    def params_for_checkpoint(self) -> Dict[str, Any]:
        # Params for init:
        all_params = {
            'hdf5_file': self.hdf5_file,
            'lazy': self.is_lazy,
            'cache_size': self.subset_cache_size,
        }

        # Subsets:
        all_params.update({
            'training set': self.training_set.params,
            'validation set': self.validation_set.params,
            'testing_set': self.testing_set.params,
        })
        return all_params

    def load_data(self, load_training=True, load_validation=True,
                  load_testing=True, subj_id: str = None,
                  volume_groups: List = None, streamline_groups: List = None):
        """
        Load raw dataset into memory.

        If `subj_id` is given, loads only this subject. Useful at tractography
        time, for instance.

        If streamline_groups is given, only counts streamlines for this group.
        With non-lazy data: only loads streamlines for this group. None means
        all. Use [] to have no volume.

        If volume_groups is given, with non-lazy data: only loads volumes for
        this group.
        """
        with h5py.File(self.hdf5_file, 'r') as hdf_handle:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            step_size = hdf_handle.attrs['step_size']
            compress = hdf_handle.attrs['compress']

            # Can't save None in hdf5, saved a string instead. Converting.
            if step_size == 'Not defined by user':
                step_size = None
            if compress == 'Not defined by user':
                compress = None

            # Loading the first training subject's group information.
            # Others should fit.
            one_subj = hdf_handle.attrs['training_subjs'][0]
            (poss_volume_groups, nb_features, poss_strea_groups,
             contains_connectivity) = prepare_groups_info(
                one_subj, hdf_handle, ref_group_info=None)
            logger.debug("Possible volume groups are: {}"
                         .format(poss_volume_groups))
            logger.debug("Number of features in each of these groups: {}"
                         .format(nb_features))
            logger.debug("Possible streamline groups are: {}"
                         .format(poss_strea_groups))
            logger.debug("Streamline groups containing a connectivity matrix: "
                         "{}".format(contains_connectivity))

            # Verifying groups of interest
            if volume_groups is not None:
                missing_vol = np.setdiff1d(volume_groups, poss_volume_groups)
                if len(missing_vol) > 0:
                    raise ValueError("Volumes {} were not found in the first "
                                     "subject of your hdf5 file."
                                     .format(missing_vol))
                vol, indv, indposs = np.intersect1d(
                    volume_groups, poss_volume_groups, return_indices=True)
                self.volume_groups = list(vol)
                self.nb_features = [nb_features[i] for i in indposs]
                logger.info("--> Chosen volume groups are: {}"
                            .format(self.volume_groups))
            else:
                logger.info("--> Using all volume groups.")
                self.volume_groups = poss_volume_groups
                self.nb_features = nb_features

            if streamline_groups is not None:
                missing_str = np.setdiff1d(streamline_groups, poss_strea_groups)
                if len(missing_str) > 0:
                    raise ValueError("Streamlines {} were not found in the "
                                     "first subject of your hdf5 file."
                                     .format(missing_str))
                self.streamline_groups, _, ind = np.intersect1d(
                    streamline_groups, poss_strea_groups, return_indices=True)
                logger.info("--> Chosen streamline groups are: {}"
                            .format(self.streamline_groups))
                self.streamlines_contain_connectivity = contains_connectivity[ind]
            else:
                logger.info("--> Using all streamline groups.")
                self.streamline_groups = poss_strea_groups
                self.streamlines_contain_connectivity = contains_connectivity

            self.streamline_groups = list(self.streamline_groups)
            group_info = (self.volume_groups, self.nb_features,
                          self.streamline_groups,
                          self.streamlines_contain_connectivity)
            self.training_set.set_subset_info(*group_info, step_size, compress)
            self.validation_set.set_subset_info(*group_info, step_size, compress)
            self.testing_set.set_subset_info(*group_info, step_size, compress)

            # LOADING
            if load_training:
                self.training_set.load(hdf_handle, subj_id)
            if load_validation:
                self.validation_set.load(hdf_handle, subj_id)
            if load_testing:
                self.testing_set.load(hdf_handle, subj_id)
