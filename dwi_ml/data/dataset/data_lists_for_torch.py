

class DataListForTorch(object):
    """
    Remembers the list of subjects and their common properties, such as the size
    of the features for the dwi volume.
    Everything is loaded into memory until it is needed.
    Will be used by multi_subjects_containers."""

    def __init__(self):
        # Feature size should be common to all subjects.
        self.subjects_data_list = []
        self.feature_size = None

    @property
    def volume_feature_size(self):
        """Returns the nb of information per voxel in the input dMRI volume."""
        if self.feature_size is None:
            try:
                self.feature_size = \
                    int(self.subjects_data_list[0].dmri_data.shape[-1])
            except IndexError:
                # No volume has been registered yet. Do not raise an exception,
                # but return 0 as the feature size.
                self.feature_size = 0
        return self.feature_size

    def add_subject(self, subject_data):
        """Adds subject's data to subjects_data_list.
        Returns idx of where subject is inserted.
        """

        # Add subject
        subject_idx = len(self.subjects_data_list)
        self.subjects_data_list.append(subject_data)

        # Make sure all volumes have the same feature size
        assert self.volume_feature_size == subject_data.dmri_data.shape[-1], \
            "Tried to add a subject whose dMRI volume's feature size was " \
            "different from previous!"
        return subject_idx

    def __getitem__(self, subject_idx):
        """ Necessary for torch"""
        return self.subjects_data_list[subject_idx]

    def __len__(self):
        return len(self.subjects_data_list)


class LazyDataListForTorch(DataListForTorch):
    def __init__(self, default_hdf_handle):
        super().__init__()
        self.hdf_handle = default_hdf_handle

    @property
    def volume_feature_size(self):
        """Overriding super's function"""
        if self.feature_size is None:
            try:
                self.feature_size = \
                    int(self.__getitem__((0, self.hdf_handle)
                                         ).dmri_data.shape[-1])
            except IndexError:
                # No volume has been registered yet. Do not raise an exception,
                # but return 0 as the feature size.
                self.feature_size = 0
        return self.feature_size

    def add_subject(self, subject_data):
        """Overriding super's function"""

        data_idx = len(self.subjects_data_list)
        self.subjects_data_list.append(subject_data)

        # Make sure all volumes have the same feature size
        new_subject = self.__getitem__((-1, self.hdf_handle))
        assert self.volume_feature_size == new_subject.dmri_data.shape[-1], \
            "Tried to add a tractogram whose dMRI volume's feature size was " \
            "different from previous!"

        return data_idx

    def __getitem__(self, subject_item):
        """Overriding super's function"""
        assert type(subject_item) == tuple, \
            "Trying to get an item, but item should be a tuple."
        subject_idx, subject_hdf_handle = subject_item
        partial_subjectdata = self.subjects_data_list[subject_idx]
        return partial_subjectdata.with_handle(subject_hdf_handle)