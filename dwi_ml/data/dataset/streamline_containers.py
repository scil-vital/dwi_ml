# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in single_subject_containers


"""
from typing import Tuple, Union, List

from dipy.io.stateful_tractogram import (set_sft_logger_level,
                                         StatefulTractogram, Space)
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

    return space_attributes, space


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
                 space_attributes: Tuple, space: Space):
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
        self.streamlines = streamlines
        self.is_lazy = None

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

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """Create an instance of this class by sending directly the hdf5 file.
        The child class's method will define how to load the data according to
        the class's data management."""
        raise NotImplementedError

    def _subset_streamlines(self, streamline_ids):
        """Return only a subset of streamlines instead of the whole tractogram.
        Useful particularly for lazy data, to avoid loading the whole data from
        the hdf5."""
        raise NotImplementedError

    def as_sft(self, streamline_ids=None):
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
                                 self.space)
        return sft


class SFTData(SFTDataAbstract):
    def __init__(self, streamlines: ArraySequence, space_attributes: Tuple,
                 space: Space, lengths_mm: List):
        super().__init__(streamlines, space_attributes, space)
        self._lengths_mm = lengths_mm
        self.is_lazy = False

    @property
    def lengths_mm(self):
        return np.array(self._lengths_mm)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        """
        Creating class instance from the hdf in cases where data is not
        loaded yet. Non-lazy = loading the data here.
        """
        streamlines = _load_streamlines_from_hdf(hdf_group)
        # Adding non-hidden parameters for nicer later access
        lengths_mm = hdf_group['euclidean_lengths']

        space_attributes, space = _load_space_from_hdf(hdf_group)

        # Return an instance of SubjectMRIData instantiated through __init__
        # with this loaded data:
        return cls(streamlines, space_attributes, space, lengths_mm)

    def _subset_streamlines(self, streamline_ids):
        if streamline_ids is not None:
            streamlines = self.streamlines[streamline_ids]
        else:
            streamlines = self.streamlines
        return streamlines


class LazySFTData(SFTDataAbstract):
    def __init__(self, streamlines: _LazyStreamlinesGetter,
                 space_attributes: Tuple, space: Space):
        super().__init__(streamlines, space_attributes, space)
        self.is_lazy = True

    @property
    def lengths_mm(self):
        # Fetching from the lazy streamline getter
        return np.array(self.streamlines.lengths_mm)

    @classmethod
    def init_from_hdf_info(cls, hdf_group: h5py.Group):
        space_attributes, space = _load_space_from_hdf(hdf_group)

        streamlines = _LazyStreamlinesGetter(hdf_group)

        return cls(streamlines, space_attributes, space)

    def _subset_streamlines(self, streamline_ids):
        streamlines = self.streamlines.get_array_sequence(streamline_ids)
        return streamlines
