# -*- coding: utf-8 -*-
import torch

from dwi_ml.data.processing.volume.interpolation import \
    torch_nearest_neighbor_interpolation, torch_trilinear_interpolation

eps = 1e-6


class TrackingMask:
    def __init__(self, dim, data=None, interp: str = 'nearest'):
        # Required dim to check if out of bounds, even with no tracking mask
        self.higher_bound = torch.as_tensor(dim[0:3])
        self.lower_bound = torch.as_tensor([0, 0, 0])[None, :]
        self.interp = interp

        if data is not None:
            assert data.shape == dim
            self.data = torch.as_tensor(data)
        else:
            self.data = None

        if interp is not None:
            assert interp in ['nearest', 'trilinear']

    def move_to(self, device):
        self.higher_bound = self.higher_bound.to(device)
        self.lower_bound = self.lower_bound.to(device)
        if self.data is not None:
            self.data = self.data.to(device)

    def is_vox_corner_in_bound(self, xyz: torch.Tensor):
        """
        xyz: Tensor
            Coordinates in voxel space, corner origin. Of shape [n, 3].
        """
        # Uses data_volume.is_coordinate_in_bound with space=VOX,
        # origin=Corner.
        xyz = torch.floor(xyz)  # vox to idx

        return ~torch.logical_or(
            torch.any(torch.less(xyz, self.lower_bound), dim=-1),
            torch.any(torch.greater_equal(xyz, self.higher_bound), dim=-1))

    def get_value_at_vox_corner_coordinate(self, xyz, interpolation):
        if interpolation == 'nearest':
            return torch_nearest_neighbor_interpolation(self.data, xyz)
        else:
            return torch_trilinear_interpolation(self.data, xyz)

    def is_in_mask(self, xyz):
        # Clipping to bound.
        xyz = torch.maximum(xyz, self.lower_bound)
        xyz = torch.minimum(xyz, self.higher_bound - eps)

        return torch.greater_equal(
            self.get_value_at_vox_corner_coordinate(xyz, self.interp),
            torch.as_tensor(0.5, device=xyz.device))
