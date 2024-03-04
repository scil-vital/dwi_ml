# -*- coding: utf-8 -*-
import os

import nibabel as nib
import numpy as np


def load_file_to4d(data_file):
    """
    Load nibabel data, and perform some checks:
    - Data must be Nifti
    - Data must be at least 3D

    Final data should be 4D (3D + features). Sending loaded data to 4D if it is
    3D, with last dimension 1.

    Returns:
    --------
    data: np.array,
    affine: np.array,
    voxel_size: np.array with size 3,
    header: nibabel header.
    """
    _, ext = os.path.splitext(data_file)

    if ext != '.gz' and ext != '.nii':
        raise ValueError('All data files should be nifti (.nii or .nii.gz) '
                         'but you provided {}. Please check again your config '
                         'file.'.format(data_file))

    img = nib.load(data_file)
    data = img.get_fdata(dtype=np.float32)

    # A note on the affine. May not be the same as when using mrinfo, because
    # nibabel flips the axes. Should be as before when saved.
    affine = img.affine
    header = img.header
    img.uncache()
    voxel_size = header.get_zooms()[:3]

    if len(data.shape) < 3:
        raise NotImplementedError('Data less than 3D is not handled.')
    elif len(data.shape) == 3:
        # Adding a fourth dimension
        data = data.reshape((*data.shape, 1))

    return data, affine, voxel_size, header
