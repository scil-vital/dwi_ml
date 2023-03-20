# -*- coding: utf-8 -*-
import numpy as np
from scilpy.image.datasets import DataVolume


class TrackingMask:
    def __init__(self, dim, data_volume: DataVolume = None):
        # Required dim to check if out of bounds, even with no tracking mask
        self.dim = dim

        # Optional tracking mask
        self.data_volume = data_volume
        if data_volume is not None:
            assert dim == data_volume.dim

    def is_vox_corner_in_bound(self, x, y, z):
        # Uses data_volume.is_coordinate_in_bound with space=VOX,
        # origin=Corner.
        i, j, k = np.floor((x, y, z))  # vox to idx
        return (0 <= i < (self.dim[0]) and
                0 <= j < (self.dim[1]) and
                0 <= k < (self.dim[2]))
