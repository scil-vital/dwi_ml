# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in single_subject_containers
"""
import logging
from typing import Tuple, Union, List

from dipy.io.stateful_tractogram import (set_sft_logger_level,
                                         StatefulTractogram, Space, Origin)
import h5py
from nibabel.streamlines import ArraySequence
import numpy as np


def _load_space_from_hdf(hdf_group: h5py.Group):
    a = np.array(hdf_group.attrs['affine'])
    d = np.array(hdf_group.attrs['dimensions'])
    vs = np.array(hdf_group.attrs['voxel_sizes'])
    vo = str(hdf_group.attrs['voxel_order'])
    space_attributes = (a, d, vs, vo)

    space_str = str(hdf_group.attrs['space']).lower()
    if space_str[0:6] == 'space.':
        # space was, ex, Space.RASMM. Changing for rasmm.
        space_str = space_str[6:]
    space = Space(space_str)

    if 'origin' in hdf_group.attrs:
        origin_str = str(hdf_group.attrs['origin']).lower()
        if origin_str[0:7] == 'origin.':
            # origin was Origin.something
            origin_str = origin_str[7:]
        if origin_str == 'trackvis':
            origin = Origin.TRACKVIS
        elif origin_str == 'nifti':
            origin = Origin.NIFTI
        else:
            origin = Origin(origin_str)
    else:
        logging.warning("Using an old hdf5! No Origin information saved. "
                        "Using dipy defaults. Make sure your SFT looks ok.")
        origin = Origin.NIFTI

    return space_attributes, space, origin


def _load_streamlines_from_hdf(hdf_group: h5py.Group):
    streamlines = ArraySequence()
    streamlines._data = np.array(hdf_group['data'])
    streamlines._offsets = np.array(hdf_group['offsets'])
    streamlines._lengths = np.array(hdf_group['lengths'])

    return streamlines


class _LazyStreamlinesGetter(object):
    def __init__(self, hdf_group):
        self.hdf_group = hdf_group

    def __getitem__(self, item):
        """
        Gets the streamlines from ids 'item', all concatenated together.
        """
        if isinstance(item, int):
            return self._get_one_streamline(item)

        if isinstance(item, list):
            streamlines = []
            for i in item:
                streamlines.append(self._get_one_streamline(i))
            return streamlines

        if isinstance(item, slice):
            streamlines = []
            offsets = self.hdf_group['offsets'][item]
            lengths = self.hdf_group['lengths'][item]
            for offset, length in zip(offsets, lengths):
                streamlines.append(
                    self.hdf_group['data'][offset:offset + length])
            return streamlines

        raise ValueError("LazyStreamlinesGetter supports only int, "
                         "list or slice")

    def _get_one_streamline(self, idx: int):
        offset = self.hdf_group['offsets'][idx]
        length = self.hdf_group['lengths'][idx]
        data = self.hdf_group['data'][offset:offset + length]

        return data

    def get_array_sequence(self, item=None):
        if item is None:
            streamlines = _load_streamlines_from_hdf(self.hdf_group)
        else:
            streamlines = ArraySequence()
            if isinstance(item, int):
                streamline = self._get_one_streamline(item)
                streamlines.append(streamline)
            elif isinstance(item, list) or isinstance(item, np.ndarray):
                for i in item:
                    streamline = self._get_one_streamline(i)
                    streamlines.append(streamline, cache_build=True)
                streamlines.finalize_append()
            elif isinstance(item, slice):
                offsets = self.hdf_group['offsets'][item]
                lengths = self.hdf_group['lengths'][item]
                for offset, length in zip(offsets, lengths):
                    streamline = self.hdf_group['data'][offset:offset + length]
                    streamlines.append(streamline, cache_build=True)
                streamlines.finalize_append()
            else:
                raise ValueError('Item should be either a int, list, '
                                 'np.ndarray or slice but we received {}'
                                 .format(type(item)))
        return streamlines

    @property
    def _lengths(self):
        """
        Get the lengths of all streamlines without loading everything into
        memory. Private to copy non-lazy SFT streamlines (ArraySequence)
        """
        lengths = np.array(self.hdf_group['lengths'], dtype=np.int16)

        return lengths

    @property
    def lengths_mm(self):
        """Get the lengths of all streamlines without loading everything
        into memory"""
        lengths = self.hdf_group['euclidean_lengths']

        return lengths

    def connectivity_matrix(self, indxyz: Tuple = None):
        if indxyz:
            indx, indy, indz = indxyz
            return np.asarray(
                self.hdf_group['connectivity_matrix'][indx, indy, indy],
                dtype=int)
        return np.asarray(self.hdf_group['connectivity_matrix'], dtype=int)

    def __len__(self):
        return len(self.hdf_group['offsets'])

    def __iter__(self):
        for i in range(len(self)):
            offset = self.hdf_group['offsets'][i]
            length = self.hdf_group['lengths'][i]
            data = self.hdf_group['data'][offset:offset + length]
            yield data


class SFTDataAbstract(object):
    """
    These classes will mimic the StatefulTractogram format, but with the
    possibility of not loading the data right away, keeping only in memory the
    hdf handle to access the data from the hdf5 file. Data in the hdf5 file is
    not a .trk format as it contains only lists and numpy arrays, but contain
    all information necessary to treat with streamlines: the data itself and
    _offset, _lengths, space attributes, etc.
    """
    def __init__(self,
                 streamlines: Union[ArraySequence, _LazyStreamlinesGetter],
                 space_attributes: Tuple, space: Space, origin: Origin,
                 contains_connectivity: bool,
                 connectivity_nb_blocs: List):
        """
        Params
        ------
        group: str
            The current streamlines group id, as loaded in the hdf5 file (it
            had type "streamlines"). Probabaly 'streamlines'.
        streamlines: ArraySequence or LazyStreamlinesGetter
            In the non-lazy version, data is the loaded data (ArraySequence).
            In the lazy version, data is the LazyStreamlinesGetter, initiated
            with the hdf_group.
        space_attributes: Tuple
            The space attributes consist of a tuple:
             (affine, dimensions, voxel_sizes, voxel_order)
        space: Space
            The space from dipy's Space format.
        subject_id: str:
            The subject's name
        """
        self.space_attributes = space_attributes
        self.space = space
        self.origin = origin
        self.streamlines = streamlines
        self.is_lazy = None
        self.contains_connectivity = contains_connectivity
        self.connectivity_nb_blocs = connectivity_nb_blocs

    @property
    def lengths(self):
        """Mimics the sft.lengths method: Returns the length (i.e. number of
        timepoints) of the streamlines."""
        # If streamlines are an ArraySequence: Accessing private info, ok.
        # If streamlines are the LazyStreamlineGetter: made lengths private to
        # mimic the ArraySequence.
        return np.array(self.streamlines._lengths, dtype=np.int16)

    @property
    def lengths_mm(self):
        """New method compared to SFTs: access to the length in mm instead of
        in terms of number of points, to facilitate work with compressed
        streamlines."""
        raise NotImplementedError

    def connectivity_matrix_and_info(self, ind=None):
        """New method compared to SFTs: access pre-computed connectivity
        matrix. Returns the subject's connectivity matrix associated with
        current tractogram, together with information required to recompute
        a similar matrix: reference volume's shape and number of blocs."""
        if not self.contains_connectivity:
            raise ValueError("No pre-computed connectivity matrix found for "
                             "this subject.")

        (_, ref_volume_shape, _, _) = self.space_attributes

        return (self._access_connectivity_matrix(ind), ref_volume_shape,
                self.connectivity_nb_blocs)

    def _access_connectivity_matrix(self, ind):
        raise NotImplementedError

    @classmethod
    def init_sft_data_from_hdf_info(cls, hdf_group: h5py.Group):
        """Create an instance of this class by sending directly the hdf5 file.
        The child class's method will define how to load the data according to
        the class's data management."""
        raise NotImplementedError

    def _subset_streamlines(self, streamline_ids):
        """Return only a subset of streamlines instead of the whole tractogram.
        Useful particularly for lazy data, to avoid loading the whole data from
        the hdf5."""
        raise NotImplementedError

    def as_sft(self, streamline_ids: List = None):
        """
        Returns chosen streamlines in a StatefulTractogram format.

        Params
        ------
        streamline_ids: list[int]
            List of chosen ids. If None, use all streamlines.
        """
        set_sft_logger_level('WARNING')
        streamlines = self._subset_streamlines(streamline_ids)

        sft = StatefulTractogram(streamlines, self.space_attributes,
                                 self.space, self.origin)

        return sft


class SFTData(SFTDataAbstract):
    streamlines: ArraySequence

    def __init__(self, lengths_mm: List, connectivity_matrix: np.ndarray,
                 **kwargs):
        super().__init__(**kwargs)
        self._lengths_mm = lengths_mm
        self._connectivity_matrix = connectivity_matrix
        self.is_lazy = False

    @property
    def lengths_mm(self):
        return np.array(self._lengths_mm)

    def _access_connectivity_matrix(self, indxyz: Tuple = None):
        if indxyz:
            indx, indy, indz = indxyz
            return self._connectivity_matrix[indx, indy, indy]
        return self._connectivity_matrix

    @classmethod
    def init_sft_data_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        streamlines = _load_streamlines_from_hdf(hdf_group)
        # Adding non-hidden parameters for nicer later access
        lengths_mm = hdf_group['euclidean_lengths']
        if 'connectivity_matrix' in hdf_group:
            contains_connectivity = True
            connectivity_matrix = np.asarray(hdf_group['connectivity_matrix'],
                                             dtype=int)
            connectivity_nb_blocs = hdf_group.attrs['connectivity_nb_blocs']
        else:
            contains_connectivity = False
            connectivity_matrix = None
            connectivity_nb_blocs = None

        space_attributes, space, origin = _load_space_from_hdf(hdf_group)

        # Return an instance of SubjectMRIData instantiated through __init__
        # with this loaded data:
        return cls(lengths_mm, connectivity_matrix,
                   streamlines=streamlines, space_attributes=space_attributes,
                   space=space, origin=origin,
                   contains_connectivity=contains_connectivity,
                   connectivity_nb_blocs=connectivity_nb_blocs)

    def _subset_streamlines(self, streamline_ids):
        if streamline_ids is not None:
            streamlines = self.streamlines[streamline_ids]
        else:
            streamlines = self.streamlines
        return streamlines


class LazySFTData(SFTDataAbstract):
    streamlines: _LazyStreamlinesGetter

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.is_lazy = True

    @property
    def lengths_mm(self):
        # Fetching from the lazy streamline getter
        return np.array(self.streamlines.lengths_mm)

    def _access_connectivity_matrix(self, indxyz: Tuple = None):
        # Fetching in a lazy way
        return self.streamlines.connectivity_matrix(indxyz)

    @classmethod
    def init_sft_data_from_hdf_info(cls, hdf_group: h5py.Group):
        space_attributes, space, origin = _load_space_from_hdf(hdf_group)
        if 'connectivity_matrix' in hdf_group:
            contains_connectivity = True
            connectivity_nb_blocs = hdf_group.attrs['connectivity_nb_blocs']
        else:
            contains_connectivity = False
            connectivity_nb_blocs = None

        streamlines = _LazyStreamlinesGetter(hdf_group)

        return cls(streamlines=streamlines, space_attributes=space_attributes,
                   space=space, origin=origin,
                   contains_connectivity=contains_connectivity,
                   connectivity_nb_blocs=connectivity_nb_blocs)

    def _subset_streamlines(self, streamline_ids):
        streamlines = self.streamlines.get_array_sequence(streamline_ids)
        return streamlines
