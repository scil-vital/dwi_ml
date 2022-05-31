# -*- coding: utf-8 -*-
import logging

import h5py
from dwi_ml.data.dataset.single_subject_containers import (SubjectDataAbstract,
                                                           SubjectData,
                                                           LazySubjectData)

logger = logging.getLogger('dataset_logger')


class SubjectsDataListAbstract(object):
    """
    Mimics a simple list, with added methods to deal with handles in the case
    of lazy data.
    """
    def __init__(self, hdf5_path: str, log):
        self.hdf5_path = hdf5_path
        self.is_lazy = None
        self._log = log

        # Do not access it directly. Use get_subj.
        # Will be a list of SubjectData or LazySubjectData
        self._subjects_data_list = []

    def add_subject(self, subject_data: SubjectDataAbstract):
        """
        Adds subject's data to subjects_data_list. Simimlar to append but also
        returns subject index.
        """
        subject_idx = len(self._subjects_data_list)
        self._subjects_data_list.append(subject_data)

        return subject_idx

    def __len__(self):
        return len(self._subjects_data_list)

    def __getitem__(self, subject_idx):
        """
        Get a specific SubjectData. In the lazy case, a handle must be
        provided, else, use open_handle_and_getitem.
        """
        raise NotImplementedError

    def open_handle_and_getitem(self, subject_idx):
        """
        Similar to __getitem__, but first opens a hdf handle with known hdf
        path.
        """
        raise NotImplementedError


class SubjectsDataList(SubjectsDataListAbstract):
    def __init__(self, hdf5_path: str, log):
        """
        The subjects_data_list will now be a list of SubjectData.
        """
        super().__init__(hdf5_path, log)
        self.is_lazy = False

    def __getitem__(self, subject_idx) -> SubjectData:
        """
        Params
        ------
        subject_item: int
            The subject idx to get.
        """
        return self._subjects_data_list[subject_idx]

    def open_handle_and_getitem(self, subject_idx):
        # Non-lazy. No need to add a handle. Simply use __getitem__
        # (indirectly).
        return self[subject_idx]


class LazySubjectsDataList(SubjectsDataListAbstract):
    def __init__(self, hdf5_path: str, log):
        """
        The subjects_data_list will now be a list of LazySubjectData.
        """
        super().__init__(hdf5_path, log)
        self.is_lazy = True

    def __getitem__(self, subject_item) -> LazySubjectData:
        """
        Params
        ------
        subject_item: Union[int, Tuple(int, hdf_handle)]
            The subject id to get [and the handle to add if necessary].
        """
        if type(subject_item) == tuple:

            subject_idx, subject_hdf_handle = subject_item
            subj_data = self._subjects_data_list[subject_idx]

            if subj_data.hdf_handle:
                if subject_hdf_handle and subj_data.hdf_handle.id.valid:
                    logger.debug("Getting item from the subjects list. You "
                                 "provided a hdf handle but subject already "
                                 "had a valid one: \n {}. Using new handle."
                                 .format(subj_data.hdf_handle))

            subj_data.add_handle(subject_hdf_handle)

        else:
            subj_data = self._subjects_data_list[subject_item]

        return subj_data

    def open_handle_and_getitem(self, subject_idx) -> LazySubjectData:
        hdf_handle = h5py.File(self.hdf5_path, 'r')
        logger.debug("Opened a new handle: {}".format(hdf_handle))

        return self[(subject_idx, hdf_handle)]
