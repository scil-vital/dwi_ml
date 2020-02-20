"""

We expect the classes here to be used in data_lists_for_torch.py
"""

import h5py
import nibabel as nib
import numpy as np
import torch


class SubjectdMRIData(object):
    """For a subject, MRI data volume with its properties such as the vox2rasmm
    affine and the subject_id.
    """

    def __init__(self, data: np.ndarray, affine_vox2rasmm: np.ndarray,
                 subject_id: str = None):
        self._data = self.get_internal_data(data)
        self.affine_vox2rasmm = torch.as_tensor(affine_vox2rasmm,
                                                dtype=torch.float)
        self.subject_id = subject_id

    @staticmethod
    def get_internal_data(data):
        return torch.as_tensor(data, dtype=torch.float)

    @property
    def as_tensor(self):
        """Returns data as a torch tensor. _data is already set as such."""
        return self._data

    @staticmethod
    def get_data_from_hdf(hdf_group: h5py.Group):
        return np.array(hdf_group['data'], dtype=np.float32)

    @classmethod
    def create_from_hdf_group(cls, hdf_group: h5py.Group):
        """Create an SubjectMRIData from an HDF group object.
        Returns an instance of this class.
        """
        # Load information from the hdf_goup.
        data = cls.get_data_from_hdf(hdf_group)
        affine_vox2rasmm = np.array(hdf_group.attrs['vox2rasmm'],
                                    dtype=np.float32)
        subject_id = hdf_group.parent.name[1:]

        # Return an instance of this class.
        return cls(data=data, affine_vox2rasmm=affine_vox2rasmm,
                   subject_id=subject_id)

    @property
    def shape(self):
        return np.array(self._data.shape)


class LazySubjectdMRIData(SubjectdMRIData):
    """Class used to encapsulate MRI metadata alongside a lazy data volume,
    such as the vox2rasmm affine or the subject_id."""

    def __init__(self, data: h5py.Group = None,
                 affine_vox2rasmm: np.ndarray = None,
                 subject_id: str = None):
        super().__init__(data, affine_vox2rasmm, subject_id)

    @staticmethod
    def get_internal_data(data):
        """Different from the non-lazy version."""
        return data

    @property
    def as_tensor(self):
        """Returns data as a torch tensor.
        Different from the non-lazy version. """
        return torch.as_tensor(np.array(self._data, dtype=np.float32),
                               dtype=torch.float)

    @staticmethod
    def get_data_from_hdf(hdf_group: h5py.Group):
        """Different from the non-lazy version."""
        return hdf_group['data']

    # toDO. Not sure if calling the super's create_from_hdf return the right
    #  type of class. Read about it and I think so but to test.


class SubjectData(object):
    """
    A "Subject" = dMRI data + streamlines + infos such as tracking mask.
    See also LazySubjectData, altough they are not parents because not similar
    at all in the way they work.
    """

    def __init__(self, dmri_data: SubjectdMRIData = None, streamlines=None,
                 lengths_mm=None):
        self.dmri_data = dmri_data
        self.streamlines = streamlines                                                      # toDo. Streamlines should be an sft???
        self.lengths_mm = lengths_mm

    @classmethod
    def create_from_hdf(cls, hdf_subject):
        """Create an instance of this class from an HDF group object."""
        dmri_data = SubjectdMRIData.create_from_hdf_group(
            hdf_subject['input_volume'])

        streamlines = None
        lengths_mm = None
        if 'streamlines' in hdf_subject:
            streamlines = nib.streamlines.ArraySequence()
            streamlines._data = np.array(hdf_subject['streamlines/data'])
            streamlines._offsets = np.array(hdf_subject['streamlines/offsets'])
            streamlines._lengths = np.array(hdf_subject['streamlines/lengths'])
            lengths_mm = np.array(hdf_subject['streamlines/euclidean_lengths'])

        return cls(dmri_data=dmri_data, streamlines=streamlines,
                   lengths_mm=lengths_mm)


class LazySubjectData(object):
    """
    A "Subject" = dMRI data + streamlines + infos such as tracking mask.
    See also SubjectData, altough they are not parents because not similar
    at all in the way they work.
    """
    def __init__(self, subject_id, hdf_handle=None):
        self.subject_id = subject_id
        self.hdf_handle = hdf_handle

    @property
    def dmri_data(self):
        """As a property, this is only computed if called by the user."""
        if self.hdf_handle is not None:
            return LazySubjectdMRIData.create_from_hdf_group(
                self.hdf_handle[self.subject_id]['input_volume'])
        else:
            return None

    @property
    def streamlines(self):
        """As a property, this is only computed if called by the user."""
        if self.hdf_handle is not None:
            return LazyStreamlinesGetter(self.hdf_handle, self.subject_id)                  # toDo. Streamlines should be an sft???
        else:
            raise ValueError("No streamlines available without an HDF handle!")

    @property
    def lengths_mm(self):
        """As a property, this is only computed if called by the user."""
        if self.hdf_handle is not None:
            name='streamlines/euclidean_lengths'
            return np.array(self.hdf_handle[self.subject_id][name])

    def with_handle(self, hdf_handle):
        return LazySubjectData(self.subject_id, hdf_handle=hdf_handle)


class LazyStreamlinesGetter(object):
                                                                                        # toDo. Streamlines should be an sft???
    def __init__(self, hdf_handle, subject_id):
        self.hdf_handle = hdf_handle
        self.subject_id = subject_id

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
            streamlines_dataset = self.hdf_handle[self.subject_id][name]
            offsets = streamlines_dataset['offsets'][item]
            lengths = streamlines_dataset['lengths'][item]
            for offset, length in zip(offsets, lengths):
                streamlines.append(
                    streamlines_dataset['data'][offset:offset + length])

        raise ValueError("LazyStreamlinesGetter supports only int, "
                         "list or slice")

    def _get_streamline(self, idx):
        streamlines_dataset = self.hdf_handle[self.subject_id]['streamlines']
        offset = streamlines_dataset['offsets'][idx]
        length = streamlines_dataset['lengths'][idx]
        data = streamlines_dataset['data'][offset:offset + length]

        return data

    def get_all(self):
        streamlines_dataset = self.hdf_handle[self.subject_id]['streamlines']

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
