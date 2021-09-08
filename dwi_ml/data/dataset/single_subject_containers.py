# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in data_list.py
"""
import logging
from typing import List, Union

from dwi_ml.data.dataset.mri_data_containers import LazyMRIData, MRIData
from dwi_ml.data.dataset.streamline_containers import LazySFTData, SFTData


class SubjectDataAbstract(object):
    """
    A "Subject" = MRI data volumes + streamlines, as in the group config used
    during the hdf5 creation.
    """
    def __init__(self, volume_groups: List[str], streamline_groups: str,
                 subject_id: str):
        """
        Parameters
        ----------
        volume_groups: List[str]
            The list of group names with type 'volume' from the config_file
            from which data was loaded.
        streamline_groups: str
            The name of the streamline group. This should be 'streamlines'.
        subject_id: str
            The subject key in the hdf file
        """
        self.volume_groups = volume_groups
        self.streamline_groups = streamline_groups
        self.subject_id = subject_id

    @classmethod
    def init_from_hdf(cls, subject_id: str, volume_groups: List[str],
                      streamline_groups: str, hdf_file, log=None):
        raise NotImplementedError

    def with_handle(self, hdf_handle):
        """This will simply return data in the non-lazy version. In the lazy
        version, this will add the hdf handle and load the data."""
        raise NotImplementedError


class SubjectData(SubjectDataAbstract):
    """Non-lazy version"""
    def __init__(self, volume_groups: List[str], streamline_groups: str,
                 subject_id: str, mri_data_list: List[MRIData] = None,
                 sft_data_list: List[SFTData] = None):
        """
        mri_data: List[SubjectMRIData]
            Volumes of MRI data in the format of SubjectMRIData classes. Each
            value of the list corresponds to a specific group from the config
            file, whose name is saved in groups.
        streamlines: nib.streamlines.ArraySequence
            The loaded streamlines. They contain ._data, ._offsets, ._lengths.
        lengths_mm: np.array
            The streamlines' euclidean lengths.
        """
        super().__init__(volume_groups, streamline_groups, subject_id)
        self.mri_data_list = mri_data_list
        self.sft_data_list = sft_data_list

    @classmethod
    def init_from_hdf(cls, subject_id: str, volume_groups: List[str],
                      streamline_groups: str, hdf_file, log=None):
        """
        Instantiating a single subject data: load info and use __init__

        When looping on all subjects in the multi_subject_container, we use
        tqdm progress bar, which does not work well with basic logger. Using
        compatible logger "log".
        """
        subject_mri_data_list = []
        subject_sft_data_list = []

        for group in volume_groups:
            log.debug('*    => Adding volume group "{}": '.format(group))
            # Creating a SubjectMRIData or a LazySubjectMRIData based on
            # lazy or non-lazy version.
            subject_mri_group_data = MRIData.init_from_hdf_info(
                hdf_file[subject_id][group])
            subject_mri_data_list.append(subject_mri_group_data)

        # Currently only one streamline group.
        for group in streamline_groups:
            log.debug("*    => Adding subject's streamlines")
            sft_data = SFTData.init_from_hdf_info(hdf_file[subject_id][group])
            subject_sft_data_list.append(sft_data)

        subj_data = cls(volume_groups, streamline_groups, subject_id,
                        mri_data_list=subject_mri_data_list,
                        sft_data_list=subject_sft_data_list)

        return subj_data

    def with_handle(self, hdf_handle):
        # data is already loaded. No need to add a handle here.
        return self


class LazySubjectData(SubjectDataAbstract):
    """
    A "Subject" = MRI data volumes from group_config + streamlines
    See also SubjectData, altough they are not parents because not similar
    at all in the way they work.
    """
    def __init__(self, volume_groups, streamline_groups, subject_id,
                 hdf_handle=None):
        super().__init__(volume_groups, streamline_groups, subject_id)
        self.hdf_handle = hdf_handle

        # Compared to non-lazy:
        # self.mri_data_list --> becomes a property
        # self.sft_data --> becomes a property
        # These properties depend on if a hdf_handle is currently used.
        # self.with_handle --> adds hdf_handle to the subject to allow loading

    @classmethod
    def init_from_hdf(cls, subject_id, volume_groups: List[str],
                      streamline_groups: str, hdf_file=None, log=None):
        """In short: this does basically nothing, the lazy data is kept
        as hdf5 file."""
        log.debug('*    => Lazy: not loading data. Saving information. Is hdf '
                  'handle none? Answer : {}'.format(hdf_file))

        return cls(volume_groups, streamline_groups, subject_id, hdf_file)

    @property
    def mri_data_list(self) -> Union[List[LazyMRIData], None]:
        """As a property, this is only computed if called by the user.
        Returns a List[LazyMRIData]"""
        if self.hdf_handle is not None:
            mri_data_list = []
            for group in self.volume_groups:
                hdf_group = self.hdf_handle[self.subject_id][group]
                mri_data_list.append(LazyMRIData.init_from_hdf_info(hdf_group))

            return mri_data_list
        else:
            logging.debug("Can't provide mri_data_list: hdf_handle not set.")
            return None

    @property
    def sft_data_list(self) -> Union[List[LazySFTData], None]:
        """As a property, this is only computed if called by the user.
        Returns a List[LazyMRIData]"""
        if self.hdf_handle is not None:
            mri_data_list = []
            for group in self.streamline_groups:
                hdf_group = self.hdf_handle[self.subject_id][group]
                mri_data_list.append(LazySFTData.init_from_hdf_info(hdf_group))

            return mri_data_list
        else:
            logging.debug("Can't provide mri_data_list: hdf_handle not set.")
            return None

    def with_handle(self, hdf_handle):
        """We could find groups directly from the subject's keys but this way
        is safer in case one subject had different keys than others. Always
        using only the wanted groups."""
        if hdf_handle is None:
            logging.warning('Using with_handle(), but hdf_handle is None!')

        return LazySubjectData(volume_groups=self.volume_groups,
                               streamline_groups=self.streamline_groups,
                               subject_id=self.subject_id,
                               hdf_handle=hdf_handle)
