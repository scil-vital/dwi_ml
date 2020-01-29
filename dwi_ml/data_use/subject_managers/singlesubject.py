"""

We expect the classes here to be used in multisubject.py
"""

import h5py
import nibabel as nib
import numpy as np
import torch


class MRIDataVolume(object):
    """Class used to encapsulate MRI metadata alongside a data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, data: np.ndarray, affine_vox2rasmm: np.ndarray,
                 subject_id: str = None):
        self.data = torch.as_tensor(data, dtype=torch.float)
        self.affine_vox2rasmm = torch.as_tensor(affine_vox2rasmm,
                                                dtype=torch.float)
        self.subject_id = subject_id

    @classmethod
    def create_from_hdf_group(cls, hdf_group: h5py.Group):
        """Create an MRIDataVolume from an HDF group object.
        Returns
        -------
        volume: MRIDataVolume
        """

        data = np.array(hdf_group['data'], dtype=np.float32)
        affine_vox2rasmm = np.array(hdf_group.attrs['vox2rasmm'],
                                    dtype=np.float32)
        subject_id = hdf_group.parent.name[1:]

        return cls(data=data, affine_vox2rasmm=affine_vox2rasmm,
                   subject_id=subject_id)

    @property
    def shape(self):
        """
        Returns
        -------
        Shape: np.ndarray
            The shape of the data.
        """
        return np.array(self.data.shape)


class LazyMRIDataVolume(object):
    """Class used to encapsulate MRI metadata alongside a lazy data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, data: h5py.Group = None,
                 affine_vox2rasmm: np.ndarray = None,
                 subject_id: str = None):
        self._data = data
        self.affine_vox2rasmm = torch.as_tensor(affine_vox2rasmm,
                                                dtype=torch.float)
        self.subject_id = subject_id

    @classmethod
    def create_from_hdf_group(cls, hdf_group: h5py.Group):
        """Create an MRIDataVolume from an HDF group object.
        Returns
        -------
        Voluem: LazyMRIDataVolume
        """

        data = hdf_group['data']
        affine_vox2rasmm = np.array(hdf_group.attrs['vox2rasmm'],
                                    dtype=np.float32)
        subject_id = hdf_group.parent.name[1:]

        return cls(data=data, affine_vox2rasmm=affine_vox2rasmm,
                   subject_id=subject_id)

    @property
    def data(self):
        """
        Returns
        -------
        data: torch.Tensor
        """
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float)

    @property
    def shape(self):
        """
        Returns
        -------
        shape: np.ndarray
            The shape of the data.
        """

        return self._data.shape


class SubjectData(object):
    """ Tractography-related data
    (input information, tracking mask, streamlines).
    """

    def __init__(self, dmri_volume=None, streamlines=None, lengths_mm=None):
        self.dmri_volume = dmri_volume
        self._streamlines = streamlines
        self.lengths_mm = lengths_mm

    @property
    def streamlines(self):

        return self._streamlines

    @classmethod
    def create_from_hdf(cls, hdf_subject):
        """Create a TractographyData object from an HDF group object."""
        dmri_volume = MRIDataVolume.create_from_hdf_group(
            hdf_subject['input_volume'])

        streamlines = None
        lengths_mm = None
        if 'streamlines' in hdf_subject:
            streamlines = nib.streamlines.ArraySequence()
            streamlines._data = np.array(hdf_subject['streamlines/data'])
            streamlines._offsets = np.array(hdf_subject['streamlines/offsets'])
            streamlines._lengths = np.array(hdf_subject['streamlines/lengths'])
            lengths_mm = np.array(hdf_subject['streamlines/euclidean_lengths'])

        return cls(dmri_volume=dmri_volume, streamlines=streamlines,
                   lengths_mm=lengths_mm)


class LazySubjectData(object):
    """Tractography-related data with lazy DataVolume loading
    (input information, tracking mask, streamlines)."""

    def __init__(self, subject_id, hdf_handle=None):
        self.subject_id = subject_id
        self._hdf_handle = hdf_handle

    @property
    def dmri_volume(self):
        if self._hdf_handle is not None:

            return LazyMRIDataVolume.create_from_hdf_group(
                self._hdf_handle[self.subject_id]['input_volume'])
        else:

            return None

    @property
    def streamlines(self):
        if self._hdf_handle is not None:
            return LazyStreamlinesGetter(self._hdf_handle, self.subject_id)
        else:
            raise ValueError("No streamlines available with an HDF handle!")

    @property
    def lengths_mm(self):
        if self._hdf_handle is not None:
            name='streamlines/euclidean_lengths'

            return np.array(self._hdf_handle[self.subject_id][name])

    def with_handle(self, hdf_handle):
        return LazySubjectData(self.subject_id, hdf_handle=hdf_handle)


class LazyStreamlinesGetter(object):
    def __init__(self, hdf_handle, subject_id):
        self._hdf_handle = hdf_handle
        self._subject_id = subject_id

    def __getitem__(self, item):
        if isinstance(item, int):
            return self._get_streamline(item)

        if isinstance(item, list):
            streamlines = []
            for i in item:
                streamlines.append(self._get_streamline(i))
            return streamlines

        if isinstance(item, slice):
            name='streamlines'
            streamlines = []
            streamlines_dataset = self._hdf_handle[self._subject_id][name]
            offsets = streamlines_dataset['offsets'][item]
            lengths = streamlines_dataset['lengths'][item]
            for offset, length in zip(offsets, lengths):
                streamlines.append(
                    streamlines_dataset['data'][offset:offset + length])

        raise ValueError("LazyStreamlinesGetter supports only int, "
                         "list or slice")

    def _get_streamline(self, idx):
        streamlines_dataset = self._hdf_handle[self._subject_id]['streamlines']
        offset = streamlines_dataset['offsets'][idx]
        length = streamlines_dataset['lengths'][idx]
        data = streamlines_dataset['data'][offset:offset + length]

        return data

    def get_all(self):
        streamlines_dataset = self._hdf_handle[self._subject_id]['streamlines']

        streamlines = nib.streamlines.ArraySequence()
        streamlines._data = np.array(streamlines_dataset['data'])
        streamlines._offsets = np.array(streamlines_dataset['offsets'])
        streamlines._lengths = np.array(streamlines_dataset['lengths'])

        return streamlines

    def get_lengths(self):
        """Get the lengths of all streamlines without loading everything
        into memory"""

        streamlines_dataset = self._hdf_handle[self._subject_id]['streamlines']
        lengths = np.array(streamlines_dataset['lengths'], dtype=np.int16)

        return lengths

    def __len__(self):
        return len(self._hdf_handle[self._subject_id]['streamlines/offsets'])

    def __iter__(self):
        streamlines_dataset = self._hdf_handle[self._subject_id]['streamlines']
        for i in range(len(self)):
            offset = streamlines_dataset['offsets'][i]
            length = streamlines_dataset['lengths'][i]
            data = streamlines_dataset['data'][offset:offset + length]
            yield data
