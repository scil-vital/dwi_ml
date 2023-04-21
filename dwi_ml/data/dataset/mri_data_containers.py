# -*- coding: utf-8 -*-
import logging
from typing import Union

import h5py
import numpy as np
import torch

from torch import Tensor

logger = logging.getLogger('dataset_logger')


class MRIDataAbstract(object):
    """
    This class is meant to be used similarly as a tensor. However, it adds the
    possibility to use the class for lazy data fetching, meaning that
    self._data does not actually contain the data but rather a handle to the
    hdf5 file containing it.

    The notion of 'volume' is not necessarily a single MRI acquisition. It
    could contain data from many "real" MRI volumes concatenated together.
    """
    def __init__(self, data: Union[torch.Tensor, h5py.Group],
                 voxres: np.ndarray, affine: np.ndarray):
        """
        Parameters
        ----------
        data: Tensor or h5py.Group
            In the non-lazy version, data is a vector of loaded data
            (np.ndarray), one per group. In the lazy version, data is the
            h5py.Group containing the data.
        voxres: np.array(3,)
            The pixel resolution, ex, using img.header.get_zooms()[:3].
        affine: np.array
            The affine.
        """
        # Data management depends on lazy or not
        self.voxres = voxres
        self.affine = affine

        # _data: in lazy, it is a hdf5 group. In non-lazy, it is the already
        # loaded data.
        self._data = data

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Allows initiating an instance of this class by sending only the
        hdf handle. This method will define how to load the data from it
        according to the child class's data organization.
        """
        raise NotImplementedError

    def convert_to_tensor(self, device) -> Tensor:
        """Returns the _data in the tensor format."""
        raise NotImplementedError

    @property
    def shape(self):
        return np.array(self._data.shape)

    @property
    def as_non_lazy(self):
        raise NotImplementedError


class MRIData(MRIDataAbstract):
    """
    In this child class, the data is a np.array containing the loaded data.
    """
    def __init__(self, data: torch.Tensor, voxres: np.ndarray,
                 affine: np.ndarray):
        super().__init__(data, voxres, affine)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        data = torch.as_tensor(np.array(hdf_group['data'], dtype=np.float32))
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        return cls(data, voxres, affine)

    def convert_to_tensor(self, device):
        # Data is already a np.array
        return self._data.to(device=device)

    @property
    def as_non_lazy(self):
        return self


class LazyMRIData(MRIDataAbstract):
    """
    In this child class, the data is a handle to the hdf5 file. Data will only
    be loaded when needed, possibly only at necessary coordinates instead of
    keeping the whole volume in memory.
    """

    def __init__(self, data: Union[h5py.Group, None], voxres: np.ndarray,
                 affine: np.ndarray):
        super().__init__(data, voxres, affine)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Not loading the data, but loading the voxres.
        """
        data = hdf_group['data']
        voxres = np.array(hdf_group.attrs['voxres'], dtype=np.float32)
        affine = np.array(hdf_group.attrs['affine'], dtype=np.float32)

        return cls(data, voxres, affine)

    # All three methods below load the data.
    # Data is not loaded yet, but sending it to a np.array will load it.

    def convert_to_tensor(self, device):
        logger.debug("Loading from hdf5 now: {}".format(self._data))
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float, device=device)

    @property
    def as_non_lazy(self):
        logger.debug("Loading from hdf5 now: {}".format(self._data))
        return MRIData(torch.as_tensor(np.array(self._data, dtype=np.float32)),
                       self.voxres, self.affine)
