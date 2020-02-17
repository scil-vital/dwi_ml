#from __future__ import annotations

import numpy as np

from scipy.ndimage import map_coordinates

B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=np.float)

idx = np.array([[0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1]], dtype=np.float)


def interpolate_volume_at_coordinates(volume: np.ndarray, coords: np.ndarray,
                                      mode: str = 'nearest'):
    """Evaluates a 3D or 4D volume data at the given coordinates using trilinear
    interpolation.

    Parameters
    ----------
    volume : np.ndarray with 3D/4D shape
        Data volume.
    coords : np.ndarray with shape (N, 3)
        3D coordinates where to evaluate the volume data.
    mode : str, optional
        Points outside the boundaries of the input are filled according to
        the given mode (‘constant’, ‘nearest’, ‘reflect’ or ‘wrap’). [‘nearest’]
        ('constant' uses 0.0 as a points outside the boundary)

    Returns
    -------
    output : np.ndarray with shape (N, #modalities)
        Values from volume.
    """

    if volume.ndim <= 2 or volume.ndim >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    if volume.ndim == 3:
        return map_coordinates(volume, coords.T, order=1, mode=mode)

    if volume.ndim == 4:
        values_4d = []
        for i in range(volume.shape[-1]):
            values_tmp = map_coordinates(volume[..., i], coords.T, order=1,
                                         mode=mode)
            values_4d.append(values_tmp)
        return np.ascontiguousarray(np.array(values_4d).T)


