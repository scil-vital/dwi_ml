# -*- coding: utf-8 -*-
"""
We expect the classes here to be used in single_subject_containers
"""
from typing import Tuple, List, Union

from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
import h5py
from nibabel.streamlines import ArraySequence
import numpy as np
from collections import defaultdict


def _load_streamlines_attributes_from_hdf(hdf_group: h5py.Group):
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

    return space_attributes, space, origin


def _load_all_streamlines_from_hdf(hdf_group: h5py.Group):
    streamlines = ArraySequence()
    streamlines._data = np.array(hdf_group['data'])
    streamlines._offsets = np.array(hdf_group['offsets'])
    streamlines._lengths = np.array(hdf_group['lengths'])

    # DPS
    hdf_dps_group = hdf_group['data_per_streamline']
    dps_dict = {}
    for dps_key in hdf_dps_group.keys():
        dps_dict[dps_key] = hdf_dps_group[dps_key][:]

    return streamlines, dps_dict


def _load_connectivity_info(hdf_group: h5py.Group):
    connectivity_nb_blocs = None
    connectivity_labels = None
    if 'connectivity_matrix' in hdf_group:
        contains_connectivity = True
        if 'connectivity_nb_blocs' in hdf_group.attrs:
            connectivity_nb_blocs = hdf_group.attrs['connectivity_nb_blocs']
        elif 'connectivity_label_volume' in hdf_group:
            connectivity_labels = np.asarray(
                hdf_group['connectivity_label_volume'], dtype=int)
        else:
            raise ValueError(
                "Information stored in the hdf5 is that it contains a "
                "connectivity matrix, but we don't know how it was "
                "created. Either 'connectivity_nb_blocs' or "
                "'connectivity_labels' should be set.")
    else:
        contains_connectivity = False
    return contains_connectivity, connectivity_nb_blocs, connectivity_labels


class _LazyStreamlinesGetter(object):
    def __init__(self, hdf_group):
        self.hdf_group = hdf_group

    def _get_one_streamline(self, idx: int):
        # Getting one value from a hdf: fast
        offset = self.hdf_group['offsets'][idx]
        length = self.hdf_group['lengths'][idx]
        data = self.hdf_group['data'][offset:offset + length]

        return data

    def _assert_dps(self, dps_dict, n_streamlines):
        for key, value in dps_dict.items():
            if len(value) != n_streamlines:
                raise ValueError(
                    f"Length of data_per_streamline {key} is {len(value)} "
                    f"but should be {n_streamlines}.")
            elif not isinstance(value, np.ndarray):
                raise ValueError(
                    f"Data_per_streamline {key} should be a numpy array, "
                    f"not a {type(value)}.")

    def get_array_sequence(self, item=None):
        if item is None:
            streamlines, data_per_streamline = _load_all_streamlines_from_hdf(
                self.hdf_group)
        else:
            streamlines = ArraySequence()
            data_per_streamline = defaultdict(list)

            # If data_per_streamline is not in the hdf5, use an empty dict
            # so that we don't add anything to the data_per_streamline in the
            # following steps.
            hdf_dps_group = self.hdf_group['data_per_streamline'] if \
                'data_per_streamline' in self.hdf_group.keys() else {}

            if isinstance(item, int):
                data = self._get_one_streamline(item)
                streamlines.append(data)

                for dps_key in hdf_dps_group.keys():
                    data_per_streamline[dps_key].append(
                        hdf_dps_group[dps_key][item])

            elif isinstance(item, list) or isinstance(item, np.ndarray):
                # Getting a list of value from a hdf5: slow. Uses fancy
                # indexing. But possible. See here:
                # https://stackoverflow.com/questions/21766145/h5py-correct-way-to-slice-array-datasets
                # Looping and accessing ourselves.
                # Good also load the whole data and access the indexes after.
                # toDo Test speed for the three options.
                for i in item:
                    data = self._get_one_streamline(i)
                    streamlines.append(data, cache_build=True)

                    for dps_key in hdf_dps_group.keys():
                        data_per_streamline[dps_key].append(
                            hdf_dps_group[dps_key][item])

                streamlines.finalize_append()

            elif isinstance(item, slice):
                offsets = self.hdf_group['offsets'][item]
                lengths = self.hdf_group['lengths'][item]
                indices = np.arange(item.start, item.stop, item.step)
                for offset, length, idx in zip(offsets, lengths, indices):
                    streamline = self.hdf_group['data'][offset:offset + length]
                    streamlines.append(streamline, cache_build=True)

                    for dps_key in hdf_dps_group.keys():
                        # Indexing with a list (e.g. [idx]) will preserve the
                        # shape of the array. Crucial for concatenation below.
                        dps_data = hdf_dps_group[dps_key][[idx]]
                        data_per_streamline[dps_key].append(dps_data)
                streamlines.finalize_append()

            else:
                raise ValueError('Item should be either a int, list, '
                                 'np.ndarray or slice but we received {}'
                                 .format(type(item)))

            # The accumulated data_per_streamline is a list of numpy arrays.
            # We need to merge them into a single numpy array so it can be
            # reused in the StatefulTractogram.
            for key in data_per_streamline.keys():
                data_per_streamline[key] = \
                    np.concatenate(data_per_streamline[key])

        self._assert_dps(data_per_streamline, len(streamlines))
        return streamlines, data_per_streamline

    @property
    def lengths(self):
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

    def __init__(self, space_attributes: Tuple, space: Space, origin: Origin,
                 contains_connectivity: bool,
                 connectivity_nb_blocs: List = None,
                 connectivity_labels: np.ndarray = None):
        """
        The lazy/non-lazy versions will have more parameters, such as the
        streamlines, the connectivity_matrix. In the case of the lazy version,
        through the LazyStreamlinesGetter.

        Parameters
        ----------
        space_attributes: Tuple
            The space attributes consist of a tuple:
             (affine, dimensions, voxel_sizes, voxel_order)
        space: Space
            The space from dipy's Space format.
        origin: Origin
            The origin from dipy's Origin format.
        contains_connectivity: bool
            If true, will search for either the connectivity_nb_blocs or the
            connectivity_from_labels information.
        connectivity_nb_blocs: List
            The information how to recreate the connectivity matrix.
        connectivity_labels: np.ndarray
            The 3D volume stating how to recreate the labels.
            (toDo: Could be managed to be lazy)
        """
        self.space_attributes = space_attributes
        self.space = space
        self.origin = origin
        self.is_lazy = None
        self.contains_connectivity = contains_connectivity
        self.connectivity_nb_blocs = connectivity_nb_blocs
        self.connectivity_labels = connectivity_labels

    def __len__(self):
        raise NotImplementedError

    @property
    def lengths(self):
        """
        Allowing accessing the lengths separately. In the lazy case, will
        allow us to sample streamlines by lengths without loading them all
        """
        raise NotImplementedError

    @property
    def lengths_mm(self):
        """
        New method compared to SFTs: access to the length in mm instead of
        in terms of number of points, to facilitate work with compressed
        streamlines. Was computed when creating the hdf5.
        """
        raise NotImplementedError

    def get_connectivity_matrix_and_info(self, ind=None):
        """New method compared to SFTs: access pre-computed connectivity
        matrix. Returns the subject's connectivity matrix associated with
        current tractogram, together with information required to recompute
        a similar matrix: reference volume's shape and number of blocs."""
        if not self.contains_connectivity:
            raise ValueError("No pre-computed connectivity matrix found for "
                             "this subject.")

        (_, ref_volume_shape, _, _) = self.space_attributes

        return (self._access_connectivity_matrix(ind), ref_volume_shape,
                self.connectivity_nb_blocs, self.connectivity_labels)

    def _access_connectivity_matrix(self, ind):
        raise NotImplementedError

    @classmethod
    def init_sft_data_from_hdf_info(cls, hdf_group: h5py.Group):
        """Create an instance of this class by sending directly the hdf5 file.
        The child class's method will define how to load the data according to
        the class's data management."""
        raise NotImplementedError

    def _get_streamlines_as_list(self, streamline_ids) -> List[ArraySequence]:
        """Return only a subset of streamlines instead of the whole tractogram.
        Useful particularly for lazy data, to avoid loading the whole data from
        the hdf5."""
        raise NotImplementedError

    def as_sft(self,
               streamline_ids: Union[List[int], int, slice, None] = None) \
            -> StatefulTractogram:
        """
        Returns chosen streamlines in a StatefulTractogram format.

        Params
        ------
        streamline_ids: Union[List[int], int, slice, None]
            List of chosen ids. If None, use all streamlines.
        """
        streamlines, dps = self._get_streamlines_as_list(streamline_ids)

        sft = StatefulTractogram(streamlines, self.space_attributes,
                                 self.space, origin=self.origin,
                                 data_per_streamline=dps)

        return sft


class SFTData(SFTDataAbstract):
    def __init__(self, streamlines: ArraySequence,
                 lengths_mm: List, connectivity_matrix: np.ndarray,
                 data_per_streamline: np.ndarray = None,
                 **kwargs):
        """
        streamlines: ArraySequence or LazyStreamlinesGetter
            In the non-lazy version, data is the loaded data (ArraySequence).
            In the lazy version, data is the LazyStreamlinesGetter, initiated
            with the hdf_group.
        """
        super().__init__(**kwargs)
        self.streamlines = streamlines
        self._lengths_mm = lengths_mm
        self._connectivity_matrix = connectivity_matrix
        self.is_lazy = False
        self.data_per_streamline = data_per_streamline

    def __len__(self):
        return len(self.streamlines)

    @property
    def lengths(self):
        """Mimics the sft.lengths method: Returns the length (i.e. number of
        timepoints) of the streamlines."""
        # If streamlines are an ArraySequence: Accessing private info, ok.
        return np.array(self.streamlines._lengths, dtype=np.int16)

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
        streamlines, dps_dict = _load_all_streamlines_from_hdf(hdf_group)
        # Adding non-hidden parameters for nicer later access
        lengths_mm = hdf_group['euclidean_lengths']

        contains_connectivity, connectivity_nb_blocs, connectivity_labels = \
            _load_connectivity_info(hdf_group)
        if contains_connectivity:
            connectivity_matrix = np.asarray(
                hdf_group['connectivity_matrix'], dtype=int)  # int or bool?
        else:
            connectivity_matrix = None

        space_attributes, space, origin = _load_streamlines_attributes_from_hdf(
            hdf_group)

        # Return an instance of SubjectMRIData instantiated through __init__
        # with this loaded data:
        return cls(streamlines=streamlines, lengths_mm=lengths_mm,
                   connectivity_matrix=connectivity_matrix,
                   space_attributes=space_attributes,
                   space=space, origin=origin,
                   contains_connectivity=contains_connectivity,
                   connectivity_nb_blocs=connectivity_nb_blocs,
                   connectivity_labels=connectivity_labels,
                   data_per_streamline=dps_dict)

    def _get_streamlines_as_list(self, streamline_ids):
        if streamline_ids is not None:
            dps_indexed = {}
            for key, value in self.data_per_streamline.items():
                dps_indexed[key] = value[streamline_ids]

            return self.streamlines.__getitem__(streamline_ids), dps_indexed
        else:
            return self.streamlines, self.data_per_streamline


class LazySFTData(SFTDataAbstract):
    def __init__(self, streamlines_getter: _LazyStreamlinesGetter, **kwargs):
        """
        streamlines_getter: LazyStreamlinesGetter
            In the lazy version, data is the LazyStreamlinesGetter, initiated
            with the hdf_group. Gives access to the streamlines, the
            space attributes, the connectivity matrix, etc.
        """
        super().__init__(**kwargs)
        self.streamlines_getter = streamlines_getter
        self.is_lazy = True

    def __len__(self):
        return len(self.streamlines_getter)

    @property
    def lengths(self):
        # Fetching from the lazy streamline getter
        return np.array(self.streamlines_getter.lengths_mm)

    @property
    def lengths_mm(self):
        # Fetching from the lazy streamline getter
        return np.array(self.streamlines_getter.lengths_mm)

    def _access_connectivity_matrix(self, indxyz: Tuple = None):
        # Fetching in a lazy way
        return self.streamlines_getter.connectivity_matrix(indxyz)

    @classmethod
    def init_sft_data_from_hdf_info(cls, hdf_group: h5py.Group):
        space_attributes, space, origin = _load_streamlines_attributes_from_hdf(
            hdf_group)

        contains_connectivity, connectivity_nb_blocs, connectivity_labels = \
            _load_connectivity_info(hdf_group)

        streamlines = _LazyStreamlinesGetter(hdf_group)

        return cls(streamlines_getter=streamlines,
                   space_attributes=space_attributes,
                   space=space, origin=origin,
                   contains_connectivity=contains_connectivity,
                   connectivity_nb_blocs=connectivity_nb_blocs,
                   connectivity_labels=connectivity_labels)

    def _get_streamlines_as_list(self, streamline_ids):
        streamlines, dps = self.streamlines_getter.get_array_sequence(
            streamline_ids)
        return streamlines, dps
