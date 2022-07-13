# -*- coding: utf-8 -*-
import logging
from typing import Union

import h5py
import numpy as np
import torch

from scilpy.image.datasets import DataVolume
from torch import Tensor

logger = logging.getLogger('dataset_logger')


class MRIDataAbstract(object):
    """
    This class is meant to be used similarly as
    scilpy.image.dtasets.DataVolume, which contains the data and all
    information necessary to use interpolation on it in any space, such as the
    voxel resolution. However, it adds the possibility to use the class
    for lazy data fetching, meaning that self.data does not actually contain
    the data but rather a handle to the hdf5 file containing it.

    The notion of 'volume' is not necessarily a single MRI acquisition. It
    could contain data from many "real" MRI volumes concatenated together.
    """
    def __init__(self, data: Union[np.ndarray, h5py.Group], voxres: np.ndarray,
                 affine: np.ndarray, interpolation: str = None):
        """
        Parameters
        ----------
        data: np.ndarray or h5py.Group
            In the non-lazy version, data is a vector of loaded data
            (np.ndarray), one per group. In the lazy version, data is the
            h5py.Group containing the data.
        voxres: np.array(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
        affine: np.array
            The affine.
        interpolation: str or None
            The interpolation choice amongst "trilinear" or "nearest". If
            None, functions getting a coordinate in mm instead of voxel
            coordinates on the data are not available.
        """
        # Data management depends on lazy or not
        self.voxres = voxres
        self.affine = affine
        self.interpolation = interpolation

        # _data: in lazy, it is a hdf5 group. In non-lazy, it is the already
        # loaded data.
        self._data = data

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        """
        Allows initiating an instance of this class by sending only the
        hdf handle. This method will define how to load the data from it
        according to the child class's data organization.
        """
        raise NotImplementedError

    @property
    def as_data_volume(self) -> DataVolume:
        """Returns the _data in the format of scilpy's DataVolume."""
        raise NotImplementedError

    @property
    def as_tensor(self) -> Tensor:
        """Returns the _data in the tensor format."""
        raise NotImplementedError

    @property
    def shape(self):
        return np.array(self._data.shape)


class MRIData(MRIDataAbstract):
    """
    In this child class, the data is a np.array containing the loaded data.
    """
    def __init__(self, data: np.ndarray, voxres: np.ndarray,
                 affine: np.ndarray, interpolation: str = None):
        super().__init__(data, voxres, affine, interpolation)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        data = np.array(hdf_group['data'], dtype=np.float32)
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        return cls(data, voxres, affine, interpolation)

    @property
    def as_data_volume(self) -> DataVolume:
        # Data is already a np.array
        return DataVolume(self._data, self.voxres, self.interpolation)

    @property
    def as_tensor(self):
        # Data is already a np.array
        return torch.as_tensor(self._data, dtype=torch.float)


class LazyMRIData(MRIDataAbstract):
    """
    In this child class, the data is a handle to the hdf5 file. Data will only
    be loaded when needed, possibly only at necessary coordinates instead of
    keeping the whole volume in memory.
    """

    def __init__(self, data: Union[h5py.Group, None], voxres: np.ndarray,
                 affine: np.ndarray, interpolation: str = None):
        super().__init__(data, voxres, affine, interpolation)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group,
                           interpolation: str = None):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Not loading the data, but loading the voxres.
        """
        data = hdf_group['data']
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        return cls(data, voxres, affine, interpolation)

    # All three methods below load the data.
    # Data is not loaded yet, but sending it to a np.array will load it.

    @property
    def as_data_volume(self) -> DataVolume:
        logger.info("LOADING FROM HDF5 NOW {}".format(self._data))
        return DataVolume(np.array(self._data, dtype=np.float32),
                          self.voxres, self.interpolation)

    @property
    def as_tensor(self):
        logger.info("LOADING FROM HDF5 NOW {}".format(self._data))
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float)

    @property
    def as_non_lazy(self):
        logger.info("LOADING FROM HDF5 NOW {}".format(self._data))
        return MRIData(np.array(self._data), self.voxres, self.affine,
                       self.interpolation)
