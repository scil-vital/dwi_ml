# -*- coding: utf-8 -*-
"""

We expect the classes here to be used in data_list.py
"""
import logging
from typing import List, Union

import h5py
import nibabel as nib
from nibabel.streamlines import ArraySequence
import numpy as np
import torch


class SubjectMRIDataAbstract(object):
    """For a subject, MRI data volume with its properties such as the vox2rasmm
    affine and the subject_id. Ex: dMRI, T1, masks, etc.
    """
    def __init__(self, group: str, data: Union[np.ndarray, h5py.Group],
                 affine: np.ndarray, subject_id: str = None):
        """
        Encapsulating data, affine and subject ID.

        Parameters
        group: str
            The current MRI's data's group id, as named in the config_file's
            group from which data was loaded (it had type 'volume').
        data: np.ndarray or h5py.Group
            In the non-lazy version, data is a vector of loaded data
            (np.ndarray), one per group. In the lazy version, data is the
            h5py.Group containing the data.
        affine_vox2rasmm: np.ndarray
            Affine is a loaded array in both the lazy and non-lazy version.
        subject_id: str:
            The subject's name
        """
        self.group = group

        # Data management depends on lazy or not
        self._data = self._get_internal_data(data)
        self.affine = torch.as_tensor(affine, dtype=torch.float)
        self.subject_id = subject_id

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group, group: str,
                           subject_id: str):
        raise NotImplementedError

    @staticmethod
    def _get_internal_data(data):
        """In non-lazy, the data is simply self._data, already loaded. In lazy,
        this returns non-loaded data."""
        raise NotImplementedError

    @property
    def as_tensor(self):
        raise NotImplementedError

    @property
    def shape(self):
        return np.array(self._data.shape)


class SubjectMRIData(SubjectMRIDataAbstract):
    def __init__(self, group: str, data: Union[np.ndarray, h5py.Group],
                 affine: np.ndarray, subject_id: str = None):
        super().__init__(group, data, affine, subject_id)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group, group: str,
                           subject_id: str):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        data = np.array(hdf_group, dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        # Return an instance of SubjectMRIData instantiated through __init__
        # with this loaded data:
        return cls(group, data, affine, subject_id)

    @staticmethod
    def _get_internal_data(data):
        return torch.as_tensor(data, dtype=torch.float)

    @property
    def as_tensor(self):
        """Returns data as a torch tensor. _data is already set as such."""
        return self._data


class LazySubjectMRIData(SubjectMRIDataAbstract):
    """Class used to encapsulate MRI metadata alongside a lazy data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, group: str, data: h5py.Group = None,
                 affine: np.ndarray = None, subject_id: str = None):
        super().__init__(group, data, affine, subject_id)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group, group: str,
                           subject_id: str):
        """Creating class instance from the hdf in cases where data is not
        loaded yet. Not loading the data, but loading the affine."""

        data = hdf_group
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        # Return an instance of LazySubjectMRIData instantiated through
        # __init__ with this non-loaded data:
        return cls(group=group, data=data, affine=affine,
                   subject_id=subject_id, )

    @staticmethod
    def _get_internal_data(data):
        """Different from the non-lazy version. Data is not loaded as a
        tensor yet."""
        return data

    @property
    def as_tensor(self):
        """Returns data as a torch tensor. Different from the non-lazy version:
        data is not a tensor yet."""
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float)


def find_group_infos(groups: List[str], hdf_subj):
    logging.debug('GROUP INFOS')
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
        subj_id: str
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
                 subject_id: str, mri_data_list: List[SubjectMRIData] = None,
                 streamlines: nib.streamlines.ArraySequence = None,
                 lengths_mm: np.array = None):
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
        self.streamlines = streamlines
        self.lengths_mm = lengths_mm

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
            log.debug('*    => Adding volume group {}:'.format(group))
            # Creating a SubjectMRIData or a LazySubjectMRIData based on
            # lazy or non-lazy version.
            subject_mri_group_data = SubjectMRIData.init_from_hdf_info(
                hdf_file[subject_id][group], group, subject_id)
            subject_mri_data_list.append(subject_mri_group_data)

        # Currently only one streamline group.
        for group in [streamline_group]:
            log.debug("*    => Adding subject's streamlines")
            streamlines = ArraySequence()
            streamlines._data = np.array(
                hdf_file[subject_id][group]['data'])
            streamlines._offsets = np.array(
                hdf_file[subject_id][group]['offsets'])
            streamlines._lengths = np.array(
                hdf_file[subject_id][group]['lengths'])
            lengths_mm = np.array(
                hdf_file[subject_id][group]['euclidean_lengths'])

        subj_data = cls(volume_groups, streamline_group, subject_id,
                        mri_data_list=subject_mri_data_list,
                        streamlines=streamlines, lengths_mm=lengths_mm)

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
        # self.streamlines --> becomes a property
        # self.lengths_mm --> becomes a property
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
    def mri_data_list(self) -> Union[List[LazySubjectMRIData], None]:
        """As a property, this is only computed if called by the user.
        Returns a List[LazySubjectMRIData],"""
        if self.hdf_handle is not None:
            if len(self.volume_groups) == 0 :
                raise NotImplementedError("Error, the mri data should not "
                                          "be loaded if the volume groups are "
                                          "not verified yet.")
            mri_data_list = []
            for group in self.volume_groups:
                hdf_group = self.hdf_handle[self.subject_id][group]
                mri_data_list.append(LazySubjectMRIData.init_from_hdf_info(
                    hdf_group, group, self.subject_id))

            return mri_data_list
        else:
            return None

    @property
    def streamlines(self):
        """As a property, this is only computed if called by the user."""
        if self.hdf_handle is not None:
            return LazyStreamlinesGetter(self.hdf_handle, self.subject_id,
                                         self.streamline_group)
        else:
            raise ValueError("No streamlines available without an HDF handle!")

    @property
    def lengths_mm(self):
        """As a property, this is only computed if called by the user."""
        if self.hdf_handle is not None:
            name = 'streamlines/euclidean_lengths'
            return np.array(self.hdf_handle[self.subject_id][name])

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


class LazyStreamlinesGetter(object):
    def __init__(self, hdf_handle, subject_id, streamline_group):
        self.hdf_handle = hdf_handle
        self.subject_id = subject_id
        self.streamline_group = streamline_group # 'streamlines'

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_streamline(item)

        if isinstance(item, list):
            streamlines = []
            for i in item:
                streamlines.append(self._get_streamline(i))
            return streamlines

        if isinstance(item, slice):
            streamlines = []
            streamlines_dataset = \
                self.hdf_handle[self.subject_id][self.streamline_group]
            offsets = streamlines_dataset['offsets'][item]
            lengths = streamlines_dataset['lengths'][item]
            for offset, length in zip(offsets, lengths):
                streamlines.append(
                    streamlines_dataset['data'][offset:offset + length])

        raise ValueError("LazyStreamlinesGetter supports only int, "
                         "list or slice")

    def _get_streamline(self, idx):
        streamlines_dataset = \
            self.hdf_handle[self.subject_id][self.streamline_group]
        offset = streamlines_dataset['offsets'][idx]
        length = streamlines_dataset['lengths'][idx]
        data = streamlines_dataset['data'][offset:offset + length]

        return data

    def get_all(self):
        streamlines_dataset = \
            self.hdf_handle[self.subject_id][self.streamline_group]

        streamlines = nib.streamlines.ArraySequence()
        streamlines._data = np.array(streamlines_dataset['data'])
        streamlines._offsets = np.array(streamlines_dataset['offsets'])
        streamlines._lengths = np.array(streamlines_dataset['lengths'])

        return streamlines

    def get_lengths(self):
        """Get the lengths of all streamlines without loading everything
        into memory"""

        streamlines_dataset = self.hdf_handle[self.subject_id]['streamlines']
        lengths = np.array(streamlines_dataset['lengths'], dtype=np.int16)

        return lengths

    def __len__(self):
        return len(self.hdf_handle[self.subject_id]['streamlines/offsets'])

    def __iter__(self):
        streamlines_dataset = self.hdf_handle[self.subject_id]['streamlines']
        for i in range(len(self)):
            offset = streamlines_dataset['offsets'][i]
            length = streamlines_dataset['lengths'][i]
            data = streamlines_dataset['data'][offset:offset + length]
            yield data
