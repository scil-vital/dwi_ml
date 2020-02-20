import logging
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import tqdm

from dwi_ml.data.dataset.single_subject_containers import LazySubjectData
from dwi_ml.cache.cache_manager import (
    SingleThreadCacheManager)
from dwi_ml.data.processing.dwi.neighbourhood import (
    get_interp_neighborhood_vectors)
from dwi_ml.data.dataset.multi_subjects_dataset import MultiSubjectDataset


class LazyMultiSubjectDataset(MultiSubjectDataset):
    """Lazy dataset containing multiple TractographyData objects."""

    def __init__(self, *args, cache_size: int = 0, **kwargs):
        # Most parameters are the same as non-lazy
        super().__init__(*args, **kwargs)

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

    def arrange_streamlines(self, subj_data, subjid, streamline_timesteps_list):
        # Find his streamlines information
        tractodata_tmp_loaded = \
            self._multisubject_manager[(tractodata_id, hdf_file)]
        n_streamlines = len(tractodata_tmp_loaded.streamlines)
        self._subjID_to_streamlineID[tractodata_id] = (
            self._total_streamlines,
            self._total_streamlines + n_streamlines)

        # Get number of timesteps per streamlines
        streamline_timesteps_list.append(
            np.array(tractodata_tmp_loaded.streamlines.get_lengths(),
                     dtype=np.int16))

    def get_subject_data(self, item):
        if self.hdf_handle is None:
            self.hdf_handle = h5py.File(self.path, 'r')
        return self._multisubject_manager[(item, self.hdf_handle)]

    def get_subject_dmri_data(self, subj_id):
        if self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                volume_data = self.volume_cache_manager[subj_id]
            except KeyError:
                # volume_data isn't cached; fetch and cache it
                volume_data = \
                    self.get_subject_data(subj_id).dmri_data.as_tensor

                # Send volume_data on device and keep it there while it's cached
                volume_data = volume_data.to(self.device)

                self.volume_cache_manager[subj_id] = volume_data
            return volume_data
        else:
            # No cache is used
            return self.get_subject_data(subj_id).dmri_data.as_tensor

    @staticmethod
    def build_data_list(hdf_file):
        assert hdf_file is not None
        return _LazyMultiSubjectDataManager(hdf_file)

    @staticmethod
    def add_subject_from_hdf(hdf_file, subject_id):
        return LazySubjectData(subject_id=subject_id)

    def __del__(self):
        if self.hdf_handle is not None:
            self.hdf_handle.close()

