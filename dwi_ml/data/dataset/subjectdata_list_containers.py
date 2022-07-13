# -*- coding: utf-8 -*-
import logging
import os

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

    def get_subj_with_handle(self, subject_idx):
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

    def get_subj_with_handle(self, subject_idx):
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
        self.hdf_handle = None

    def __getitem__(self, subject_item) -> LazySubjectData:
        """
        Params
        ------
        subject_item: Union[int, Tuple(int, hdf_handle)]
            If it is int or (int, None): return the LazySubjectData for given
            subject.
            If it is (int, hdf_handle): add handle to the LazySubjectData.

        Returns
        -------
        subj: LazySubjectData
            Subject as is (loaded or not).
        """
        if isinstance(subject_item, tuple):
            raise DeprecationWarning("Deprecated use. Simply add "
                                     "subj.add_handle to your getitem(idx).")

        return self._subjects_data_list[subject_item]

    def get_subj_with_handle(self, subject_idx) -> LazySubjectData:
        """
        getitem but open an handle first (if none existed).

        Params
        ------
        subject_idx: int
            Subject index.
        new_hdf_handle: hdf handle or None
            If None, use list's self.handle, or open a new one if not valid.
            If it is not None, ignore self.handle and use provided handle
            instead.

        Returns
        -------
        subj: LazySubjectData
        """
        subj_data = self.__getitem__(subject_idx)

        logger.debug("Giving handle information to subject.")

        # Verify current handle.
        if not self.hdf_handle or not self.hdf_handle.id.valid:
            if self.hdf_handle:
                self.hdf_handle.close()
            self.hdf_handle = h5py.File(self.hdf5_path, 'r')
            logger.debug("PROCESS ID {}: Opened a new handle: {}"
                         .format(os.getpid(), self.hdf_handle))
        subj_data.add_handle(self.hdf_handle)

        return subj_data
