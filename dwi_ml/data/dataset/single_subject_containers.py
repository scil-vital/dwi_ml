# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in data_list.py
"""
import logging
from typing import List, Union

from dwi_ml.data.dataset.mri_data_containers import LazyMRIData, MRIData
from dwi_ml.data.dataset.streamline_containers import LazySFTData, SFTData
from dwi_ml.data.dataset.utils import prepare_groups_info


class SubjectDataAbstract(object):
    """
    A "Subject" = MRI data volumes + streamlines, as in the group config used
    during the hdf5 creation.
    """
    def __init__(self, volume_groups: List[str], nb_features: List[int],
                 streamline_groups: List[str], subject_id: str):
        """
        Parameters
        ----------
        volume_groups: List[str]
            The list of group names with type 'volume' from the config_file
            from which data was loaded.
        nb_features: List[int]
            The number of expected feature per group.
        streamline_groups: List[str]
            The name of the streamline group. This should be 'streamlines'.
        subject_id: str
            The subject key in the hdf file
        """
        self.volume_groups = volume_groups
        self.nb_features = nb_features
        self.streamline_groups = streamline_groups
        self.subject_id = subject_id
        self.is_lazy = None

    @property
    def mri_data_list(self):
        raise NotImplementedError

    @property
    def sft_data_list(self):
        raise NotImplementedError

    @classmethod
    def init_from_hdf(cls, subject_id: str, log, hdf_file, group_info=None):
        raise NotImplementedError

    def add_handle(self, hdf_handle):
        raise NotImplementedError


class SubjectData(SubjectDataAbstract):
    """Non-lazy version"""
    def __init__(self, subject_id: str, volume_groups: List[str],
                 nb_features: List[int], mri_data_list: List[MRIData] = None,
                 streamline_groups: List[str] = None,
                 sft_data_list: List[SFTData] = None):
        """
        Additional params compared to super:
        sft_data_list: List[SFTData]
            The loaded streamlines in a format copying the SFT. They contain
            ._data, ._offsets, ._lengths, ._lengths_mm.
        """
        super().__init__(volume_groups, nb_features, streamline_groups,
                         subject_id)
        self._mri_data_list = mri_data_list
        self._sft_data_list = sft_data_list
        self.is_lazy = False

    @property
    def mri_data_list(self):
        return self._mri_data_list

    @property
    def sft_data_list(self):
        return self._sft_data_list

    @classmethod
    def init_from_hdf(cls, subject_id: str, log, hdf_file, group_info=None):
        """
        Instantiating a single subject data: load info and use __init__

        When looping on all subjects in the multi_subject_container, we use
        tqdm progress bar, which does not work well with basic logger. Using
        compatible logger "log".
        """
        volume_groups, nb_features, streamline_groups = prepare_groups_info(
            subject_id, log, hdf_file, group_info)

        subject_mri_data_list = []
        subject_sft_data_list = []

        for group in volume_groups:
            log.debug('        Loading volume group "{}": '.format(group))
            # Creating a SubjectMRIData or a LazySubjectMRIData based on
            # lazy or non-lazy version.
            subject_mri_group_data = MRIData.init_from_hdf_info(
                hdf_file[subject_id][group])
            subject_mri_data_list.append(subject_mri_group_data)

        # Currently only one streamline group.
        for group in streamline_groups:
            log.debug("        Loading subject's streamlines")
            sft_data = SFTData.init_from_hdf_info(hdf_file[subject_id][group])
            subject_sft_data_list.append(sft_data)

        subj_data = cls(subject_id,
                        volume_groups, nb_features, subject_mri_data_list,
                        streamline_groups, subject_sft_data_list)

        return subj_data

    def add_handle(self, hdf_handle):
        pass


class LazySubjectData(SubjectDataAbstract):
    """
    A "Subject" = MRI data volumes from group_config + streamlines
    See also SubjectData, altough they are not parents because not similar
    at all in the way they work.
    """
    def __init__(self, volume_groups: List[str], nb_features: List[int],
                 streamline_groups: List[str], subject_id: str,
                 hdf_handle=None):
        """
        Additional params compared to super:
        hdf_handle:
            Opened hdf file, if any. If None, data loading is deactivated.
        """
        super().__init__(volume_groups, nb_features, streamline_groups,
                         subject_id)
        self.hdf_handle = hdf_handle

    @classmethod
    def init_from_hdf(cls, subject_id: str, log, hdf_file, group_info=None):
        """
        Instantiating a single subject data: NOT LOADING info and use __init__
        (so in short: this does basically nothing, the lazy data is kept
        as hdf5 file.

        Parameters
        ----------
        subject_id: str
            Name of the subject
        log: logging.Logger
            Logger. Useful to use tqdm-compatible logger if instantiating
            subjects in a tqdm loop.
        hdf_file: h5py.File
            hdf handle.
        group_info: Tuple(str, int, str)
            Tuple containing (volume_groups, nb_features, streamline_groups)
            for this subject.
        """
        volume_groups, nb_features, streamline_groups = prepare_groups_info(
            subject_id, log, hdf_file, group_info)

        log.debug('     Lazy: not loading data.')

        return cls(volume_groups, nb_features, streamline_groups, subject_id,
                   hdf_handle=None)

    @property
    def mri_data_list(self) -> Union[List[LazyMRIData], None]:
        """As a property, this is only computed if called by the user.
        Returns a List[LazyMRIData]"""

        if self.hdf_handle is not None:
            if not self.hdf_handle.id.valid:
                logging.warning("Tried to access subject's volumes but its "
                                "hdf handle is not valid (closed file?)")
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
            sft_data_list = []
            for group in self.streamline_groups:
                hdf_group = self.hdf_handle[self.subject_id][group]
                sft_data_list.append(LazySFTData.init_from_hdf_info(hdf_group))

            return sft_data_list
        else:
            logging.debug("Can't provide sft_data_list: hdf_handle not set.")
        return None

    def add_handle(self, hdf_handle):
        """We could find groups directly from the subject's keys but this way
        is safer in case one subject had different keys than others. Always
        using only the wanted groups."""
        if hdf_handle is None or not hdf_handle.id.valid:
            logging.warning('Adding handle(), but provided handle is None or'
                            'is non valid (probably closed)!')

        self.hdf_handle = hdf_handle

    def __del__(self):
        if self.hdf_handle is not None:
            self.hdf_handle.close()
