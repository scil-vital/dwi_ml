# -*- coding: utf-8 -*-
import logging

import h5py
from dwi_ml.data.dataset.single_subject_containers import (SubjectDataAbstract,
                                                           SubjectData,
                                                           LazySubjectData)


class SubjectsDataListAbstract(object):
    """
    Remembers the list of subjects and their common properties, such as the
    size of the features for the dwi volumes.
    Everything is loaded into memory until it is needed.
    Will be used by multi_subjects_containers.
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
        Adds subject's data to subjects_data_list.
        Returns idx of where subject is inserted.
        """
        subject_idx = len(self._subjects_data_list)
        self._subjects_data_list.append(subject_data)

        return subject_idx

    def __len__(self):
        return len(self._subjects_data_list)

    def __getitem__(self, subject_idx):
        raise NotImplementedError

    def open_handle_and_getitem(self, subject_idx):
        raise NotImplementedError


class SubjectsDataList(SubjectsDataListAbstract):
    def __init__(self, hdf5_path: str, log):
        """
        The subjects_data_list will now be a list of SubjectData
        """
        super().__init__(hdf5_path, log)
        self.is_lazy = False

    def __getitem__(self, subject_idx) -> SubjectData:
        """ Necessary for torch"""
        return self._subjects_data_list[subject_idx]

    def open_handle_and_getitem(self, subject_idx):
        # Non-lazy. No need to add a handle.
        return self[subject_idx]


class LazySubjectsDataList(SubjectsDataListAbstract):
    def __init__(self, hdf5_path: str, log):
        """
        The subjects_data_list will now be a list of LazySubjectData
        """
        super().__init__(hdf5_path, log)
        self.is_lazy = True

    def __getitem__(self, subject_item) -> LazySubjectData:
        """
        Getting an item (i.e. loading).
        Params
        ------
        subject_item: Tuple(int, hdf_handle)
            The subject id to get and the handle to add if necessary.
        """
        assert type(subject_item) == tuple, \
            "Lazy SubjectsDataList: Trying to get an item, but item should " \
            "be a tuple: (subj_idx, hdf_handle)"

        subject_idx, subject_hdf_handle = subject_item
        subj_data = self._subjects_data_list[subject_idx]

        if subj_data.hdf_handle:
            if subject_hdf_handle and subj_data.hdf_handle.id.valid:
                logging.debug("Getting item from the subjects list. You "
                              "provided a hdf handle but subject already had "
                              "a valid one: \n {}. Using new handle."
                              .format(subj_data.hdf_handle))

        subj_data.add_handle(subject_hdf_handle)

        return subj_data

    def open_handle_and_getitem(self, subject_idx) -> LazySubjectData:
        """
        Same as get item, but instead of using provided handle, we open a new
        one.
        """
        hdf_handle = h5py.File(self.hdf5_path, 'r')
        logging.debug("Opened a new handle: {}".format(hdf_handle))

        return self[(subject_idx, hdf_handle)]
