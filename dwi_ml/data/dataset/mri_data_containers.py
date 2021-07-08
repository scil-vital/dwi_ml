# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in single_subject_containers.

These classes will mimic the Nifti1Image format, but loading data from the
hdf5 (which contains only lists and numpy arrays) instead of loading data from
.nii files. They will contain all information necessary to treat with MRI-like
data: the data itself and the affine.
"""

from typing import Union

import h5py
import numpy as np
import torch


class MRIDataAbstract(object):
    """For a subject, MRI data volume with its properties such as the vox2rasmm
    affine and the subject_id. Ex: dMRI, T1, masks, etc.
    """
    def __init__(self, data: Union[np.ndarray, h5py.Group],
                 affine: np.ndarray):
        """
        Encapsulating data, affine and subject ID.

        Parameters
        data: np.ndarray or h5py.Group
            In the non-lazy version, data is a vector of loaded data
            (np.ndarray), one per group. In the lazy version, data is the
            h5py.Group containing the data.
        affine_vox2rasmm: np.ndarray
            Affine is a loaded array in both the lazy and non-lazy version.
        """
        # Data management depends on lazy or not
        self._data = self._get_internal_data(data)
        self.affine = torch.as_tensor(affine, dtype=torch.float)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
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


class MRIData(MRIDataAbstract):
    def __init__(self, data: np.ndarray, affine: np.ndarray):
        super().__init__(data, affine)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        data = np.array(hdf_group['data'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        # Return an instance of SubjectMRIData instantiated through __init__
        # with this loaded data:
        return cls(data, affine)

    @staticmethod
    def _get_internal_data(data):
        return torch.as_tensor(data, dtype=torch.float)

    @property
    def as_tensor(self):
        """Returns data as a torch tensor. _data is already set as such."""
        return self._data


class LazyMRIData(MRIDataAbstract):
    """Class used to encapsulate MRI metadata alongside a lazy data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, data: h5py.Group = None, affine: np.ndarray = None):
        super().__init__(data, affine,)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """Creating class instance from the hdf in cases where data is not
        loaded yet. Not loading the data, but loading the affine."""

        data = hdf_group['data']
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        # Return an instance of LazySubjectMRIData instantiated through
        # __init__ with this non-loaded data:
        return cls(data, affine)

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
