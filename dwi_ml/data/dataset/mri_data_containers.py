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

from scilpy.image.datasets import DataVolume
from torch import Tensor


class MRIDataAbstract(object):
    """For a subject, MRI data volume with its properties such as the vox2rasmm
    affine and the subject_id. Ex: dMRI, T1, masks, etc.
    """
    def __init__(self, data: Union[np.ndarray, h5py.Group],
                 affine: np.ndarray, voxres: np.ndarray,
                 interpolation: str = None):
        """
        Encapsulating data, affine and subject ID.

        Parameters
        ----------
        data: np.ndarray or h5py.Group
            In the non-lazy version, data is a vector of loaded data
            (np.ndarray), one per group. In the lazy version, data is the
            h5py.Group containing the data.
        affine: np.ndarray
            Affine is a loaded array in both the lazy and non-lazy version.
        voxres: np.array(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
        interpolation: str or None
            The interpolation choice amongst "trilinear" or "nearest". If
            None, functions getting a coordinate in mm instead of voxel
            coordinates on the data are not available.
        """
        # Data management depends on lazy or not
        self.voxres = voxres
        self.interpolation = interpolation
        self.affine = torch.as_tensor(affine, dtype=torch.float)

        # _data: in lazy, it is a hdf5 group. In non-lazy, it is the already
        # loaded data.
        self._data = data

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        raise NotImplementedError

    @property
    def as_data_volume(self) -> DataVolume:
        raise NotImplementedError

    @property
    def as_tensor(self) -> Tensor:
        raise NotImplementedError

    @property
    def shape(self):
        return np.array(self._data.shape)


class MRIData(MRIDataAbstract):
    def __init__(self, data: np.ndarray, affine: np.ndarray,
                 voxres: np.ndarray, interpolation: str = None):
        super().__init__(data, affine, voxres, interpolation)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        data = np.array(hdf_group['data'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)

        return cls(data, affine, voxres, interpolation)

    @property
    def as_data_volume(self) -> DataVolume:
        """Data is a np array"""
        return DataVolume(self._data, self.voxres, self.interpolation)

    @property
    def as_tensor(self):
        """Returns data as a torch tensor."""
        return torch.as_tensor(self._data, dtype=torch.float)


class LazyMRIData(MRIDataAbstract):
    """Class used to encapsulate MRI metadata alongside a lazy data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, data: h5py.Group = None, affine: np.ndarray = None,
                 voxres: np.ndarray = None, interpolation: str = None):
        super().__init__(data, affine, voxres, interpolation)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Not loading the data, but loading the affine and res.
        """
        data = hdf_group['data']
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)

        return cls(data, affine, voxres, interpolation)

    @property
    def as_data_volume(self) -> DataVolume:
        """Data is a np array"""
        return DataVolume(np.array(self._data, dtype=np.float32),
                          self.voxres, self.interpolation)

    @property
    def as_tensor(self):
        """Returns data as a torch tensor. Different from the non-lazy version:
        data is not loaded yet."""
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float)
