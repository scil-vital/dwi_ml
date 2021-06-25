# -*- coding: utf-8 -*-
import logging

from dwi_ml.data.dataset.single_subject_containers import (SubjectDataAbstract,
                                                           SubjectData,
                                                           LazySubjectData)


class DataListForTorchAbstract(object):
    """
    Remembers the list of subjects and their common properties, such as the
    size of the features for the dwi volumes.
    Everything is loaded into memory until it is needed.
    Will be used by multi_subjects_containers.
    """
    def __init__(self):
        # Feature sizes should be common to all subjects.
        self.subjects_data_list = []  # List of SubjectData
        self.feature_sizes = []  # One value per volume group.
        self.groups = []  # Will be set by the first subj. Others must fit.

    def set_feature_sizes(self):
        """
        Sets the number of information per voxel in each group's dMRI volume.
        """
        if len(self.feature_sizes) > 0:
            logging.warning("You have already set the feature sizes based on "
                            "the first subject! Doing it again but you should "
                            "verify this code!")
        if len(self.subjects_data_list) == 0:
            raise ValueError('First subject must be added to the list before '
                             'verifying its feature sizes.')

        for i in range(len(self.groups)):
            self.feature_sizes.append(self._get_group_feature_size(0, i))

    def _get_group_feature_size(self, subj: int, group: int):
        """Get subject #subj's group #group's feature size"""
        raise NotImplementedError

    def add_subject(self, subject_data: SubjectDataAbstract):
        """
        Adds subject's data to subjects_data_list.
        Returns idx of where subject is inserted.
        """
        subject_idx = len(self.subjects_data_list)
        self.subjects_data_list.append(subject_data)

        # Make sure all volumes are there and have the same feature sizes
        if subject_idx == 0:
            # Set values from the first subject
            assert len(self.groups) == 0
            self.groups = subject_data.volume_groups
            self.set_feature_sizes()
        else:
            # Make sure we have the right number of groups
            if len(subject_data.volume_groups) != len(self.groups):
                raise ValueError("Tried to add a subjects who had a different "
                                 "number of volume groups than previous!")

            for i in range(len(self.groups)):
                group_size = self._get_group_feature_size(subject_idx, i)
                if self.feature_sizes != group_size:
                    raise ValueError(
                        "Tried to add a subject whose dMRI volume's feature "
                        "size was different from previous! Previous: {}, "
                        "current: {}".format(self.feature_sizes, group_size))
        return subject_idx

    def __len__(self):
        return len(self.subjects_data_list)

    def __getitem__(self, subject_idx):
        raise NotImplementedError


class DataListForTorch(DataListForTorchAbstract):

    def __init__(self):
        super().__init__()

    def _get_group_feature_size(self, subj: int, group: int):
        s = self.subjects_data_list[subj].mri_data_list[group].shape[-1]
        return int(s)

    def __getitem__(self, subject_idx):
        """ Necessary for torch"""
        return self.subjects_data_list[subject_idx]


class LazyDataListForTorch(DataListForTorch):
    def __init__(self, default_hdf_handle):
        super().__init__()
        self.hdf_handle = default_hdf_handle

    def _get_group_feature_size(self, subj: int, group: int):
        # Uses __getitem__ which means that a handle will be used.
        return int(self.__getitem__((subj, self.hdf_handle)
                                    ).mri_data_list[group].shape[-1])

    def __getitem__(self, subject_item) -> LazySubjectData:
        assert type(subject_item) == tuple, \
            "Trying to get an item, but item should be a tuple: " \
            "(subj_idx, hdf_handle, groups) where groups is the list of " \
            "groups to load for this subject."

        subject_idx, subject_hdf_handle, groups = subject_item
        partial_subjectdata = self.subjects_data_list[subject_idx]
        subj_with_handle = partial_subjectdata.with_handle(subject_hdf_handle,
                                                           groups)
        return subj_with_handle
