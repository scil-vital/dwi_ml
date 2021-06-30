# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in data_list.py
"""
import logging
from typing import List, Union

from dwi_ml.data.dataset.mri_data_containers import LazyMRIData, MRIData
from dwi_ml.data.dataset.streamline_containers import LazySFTData, SFTData


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


class SubjectDataAbstract(object):
    """
    A "Subject" = MRI data volumes from group_config + streamlines.
    See also LazySubjectData, altough they are not parents because not similar
    at all in the way they work.
    """
    def __init__(self, volume_groups: List[str], streamline_group: str,
                 subject_id: str):
        """
        Parameters
        ----------
        volume_groups: List[str]
            The list of group names with type 'volume' from the config_file
            from which data was loaded.
        streamline_group: str
            The name of the streamline group. This should be 'streamlines'.
        subject_id: str
            The subject key in the hdf file
        """
        self.volume_groups = volume_groups
        self.streamline_group = streamline_group
        self.subject_id = subject_id

    @classmethod
    def init_from_hdf(cls, subject_id: str, groups: List[str],
                      hdf_file, log=None):
        raise NotImplementedError

    def with_handle(self, hdf_handle, groups):
        """This will simply return data in the non-lazy version. In the lazy
        version, this will add the hdf handle and load the data."""
        raise NotImplementedError


class SubjectData(SubjectDataAbstract):
    """Non-lazy version"""
    def __init__(self, volume_groups: List[str], streamline_group: str,
                 subject_id: str, mri_data_list: List[MRIData] = None,
                 sft_data: SFTData = None):
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
        super().__init__(volume_groups, streamline_group, subject_id)
        self.mri_data_list = mri_data_list
        self.sft_data = sft_data

    @classmethod
    def init_from_hdf(cls, subject_id: str, groups: List[str],
                      hdf_file, log=None):
        """
        Instantiating a single subject data: load info and use __init__

        When looping on all subjects in the multi_subject_container, we use
        tqdm progress bar, which does not work well with basic logger. Using
        compatible logger "log".
        """
        subject_mri_data_list = []

        (volume_groups, streamline_group) = find_group_infos(
            groups, hdf_file[subject_id])

        for group in volume_groups:
            log.debug('*    => Adding volume group "{}": '.format(group))
            # Creating a SubjectMRIData or a LazySubjectMRIData based on
            # lazy or non-lazy version.
            subject_mri_group_data = MRIData.init_from_hdf_info(
                hdf_file[subject_id][group])
            subject_mri_data_list.append(subject_mri_group_data)

        # Currently only one streamline group.
        for group in [streamline_group]:
            log.debug("*    => Adding subject's streamlines")
            subject_sft_data = SFTData.init_from_hdf_info(
                hdf_file[subject_id][group])

        subj_data = cls(volume_groups, streamline_group, subject_id,
                        mri_data_list=subject_mri_data_list,
                        sft_data=subject_sft_data)

        return subj_data

    def with_handle(self, hdf_handle, groups):
        # data is already loaded. No need to add a handle here.
        return self


class LazySubjectData(SubjectDataAbstract):
    """
    A "Subject" = MRI data volumes from group_config + streamlines
    See also SubjectData, altough they are not parents because not similar
    at all in the way they work.
    """
    def __init__(self, volume_groups, streamline_group, subject_id,
                 hdf_handle=None):
        super().__init__(volume_groups, streamline_group, subject_id)
        self.hdf_handle = hdf_handle

        # Compared to non-lazy:
        # self.mri_data --> becomes a property
        # self.sft_data --> becomes a property
        # These properties depend on if a hdf_handle is currently used.
        # self.with_handle --> adds hdf_handle to the subject to allow loading

    @classmethod
    def init_from_hdf(cls, subject_id, groups, hdf_file=None, log=None):
        """In short: this does basically nothing, the lazy data is kept
        as hdf5 file."""

        if hdf_file:
            (volume_groups, streamline_group) = find_group_infos(
                groups, hdf_file[subject_id])
        else:
            volume_groups = []
            streamline_group = None

        return cls(volume_groups, streamline_group, subject_id, hdf_file)

    @property
    def mri_data_list(self) -> Union[List[LazyMRIData], None]:
        """As a property, this is only computed if called by the user.
        Returns a List[LazyMRIData]"""
        if self.hdf_handle is not None:
            if len(self.volume_groups) == 0:
                raise NotImplementedError("Error, the mri data should not "
                                          "be loaded if the volume groups are "
                                          "not verified yet.")
            mri_data_list = []
            for group in self.volume_groups:
                hdf_group = self.hdf_handle[self.subject_id][group]
                mri_data_list.append(LazyMRIData.init_from_hdf_info(hdf_group))

            return mri_data_list
        else:
            return None

    @property
    def sft_data(self) -> LazySFTData:
        """As a property, this is only computed if called by the user.
        Returns a LazySFTData,"""
        if self.hdf_handle is not None:
            # Mutiple streamline groups not implemented. Could be
            # for group in self.streamline_group
            hdf_group = self.hdf_handle[self.subject_id][self.streamline_group]
            return LazySFTData.init_from_hdf_info(hdf_group)

    def with_handle(self, hdf_handle, groups):
        """We could find groups directly from the subject's keys but this way
        is safer in case one subject had different keys than others. Always
        using only the wanted groups."""
        if hdf_handle is None:
            logging.warning('Using with_handle(), but hdf_handle is None!')

        if len(self.volume_groups) == 0:
            first_subj_id = list(hdf_handle.keys())[0]
            (self.volume_groups, self.streamline_group) = find_group_infos(
                groups, hdf_handle[first_subj_id])

        return LazySubjectData(volume_groups=self.volume_groups,
                               streamline_group=self.streamline_group,
                               subject_id=self.subject_id,
                               hdf_handle=hdf_handle)

