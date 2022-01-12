# -*- coding: utf-8 -*-
import torch
import numpy as np
from scipy.ndimage import map_coordinates

from dwi_ml.data.processing.space.neighborhood import \
    extend_coordinates_with_neighborhood

B1 = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
               [-1, 0, 0, 0, 1, 0, 0, 0],
               [-1, 0, 1, 0, 0, 0, 0, 0],
               [-1, 1, 0, 0, 0, 0, 0, 0],
               [1, 0, -1, 0, -1, 0, 1, 0],
               [1, -1, -1, 1, 0, 0, 0, 0],
               [1, -1, 0, 0, -1, 1, 0, 0],
               [-1, 1, 1, -1, 1, -1, -1, 1]], dtype=np.float)

# We will use the 8 voxels surrounding current position to interpolate a
# value. See ref https://spie.org/samples/PM159.pdf. The point p000 = [0, 0, 0]
# is the bottom corner of the current position (using floor).
idx_box = np.array([[0, 0, 0],
                    [0, 0, 1],
                    [0, 1, 0],
                    [0, 1, 1],
                    [1, 0, 0],
                    [1, 0, 1],
                    [1, 1, 0],
                    [1, 1, 1]], dtype=np.float)


def torch_trilinear_interpolation(volume: torch.Tensor, coords: torch.Tensor):
    """Evaluates the data volume at given coordinates using trilinear
    interpolation on a torch tensor.

    Interpolation is done using the device on which the volume is stored.

    * Note. There is a function in torch:
    torch.nn.functional.interpolation with mode trilinear
    But it resamples volumes, not coordinates.

    Parameters
    ----------
    volume : torch.Tensor with 3D or 4D shape
        The input volume to interpolate from
    coords : torch.Tensor with shape (N,3)
        The coordinates where to interpolate. (Origin = corner, space = vox).

    Returns
    -------
    output : torch.Tensor with shape (N, #modalities)
        The list of interpolated values
    coords_to_idx_clipped: the coords after floor and clipping in box.

    References
    ----------
    [1] https://spie.org/samples/PM159.pdf
    """
    # Get device, and make sure volume and coords are using the same one
    assert volume.device == coords.device,\
        "volume on device: {}; " \
        "coords on device: {}".format(volume.device, coords.device)
    device = volume.device

    # Send data to device
    idx_box_torch = torch.as_tensor(idx_box, dtype=torch.float, device=device)
    B1_torch = torch.as_tensor(B1, dtype=torch.float, device=device)

    if volume.dim() <= 2 or volume.dim() >= 5:
        raise ValueError("Volume must be 3D or 4D!")

    # indices are the floor of coordinates + idx, boxes with 8 corners around
    # given coordinates. (Floor means origin = corner)
    # coords' shape = [n_timesteps, 3]
    # coords[:, None, :] shape = [n_timesteps, 3]
    # coords[:, None, :] + idx_torch shape: [n_timesteps, 8, 3]
    #   -> the box around each time step
    # reshaped as (-1,3) = [n_timesteps*8, 3]
    # torch needs indices to be cast to long
    idx_box_unclipped = \
        torch.floor(coords[:, None, :] + idx_box_torch).reshape((-1, 3)).long()

    # Clip indices to make sure we don't go out-of-bounds
    # Origin = corner means the minimum is 0.
    #                       the maximum is shape.
    # Ex, for shape 150, last voxel is #149, with possible coords up to 149.99.
    lower = torch.as_tensor([0, 0, 0], device=device)
    upper = torch.as_tensor(volume.shape[:3], device=device) - 1
    idx_box_clipped = torch.min(torch.max(idx_box_unclipped, lower), upper)

    # Setting Q1 such as in equation 9.9
    d = coords - torch.floor(coords)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                      dx * dy, dy * dz, dx * dz,
                      dx * dy * dz], dim=0)

    if volume.dim() == 3:
        # Fetch volume data at indices based on equation 9.11.
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2]]
        p = p.reshape((coords.shape[0], -1)).t()  # -1 = the 8 corners

        # Finding coordinates with equation 9.12a.
        output = torch.sum(p * torch.mm(B1_torch.t(), Q1), dim=0)

        return output
    elif volume.dim() == 4:

        # Fetch volume data at indices
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2], :]
        p = p.reshape((coords.shape[0], 8, volume.shape[-1]))

        output = \
            torch.sum(p * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output
    else:
        raise ValueError("Interpolation: There was a problem with the "
                         "volume's number of dimensions!")


def interpolate_volume_in_neighborhood(
        volume_as_tensor, coords, neighborhood_points=None,
        device=torch.device('cpu')):
    """
    Params
    ------
    data_tensor: tensor
        The data.
    coords: a np.array(n, 3)
        One set of 3D coordinates per neighborhood to compute.
        Neighborhood will be added to these points based on model
        parameters.
        Coords must be in voxel world, origin='corner', to use
        trilinear interpolation.
    neighborhood_points:
        The neighboors to add to each coord. If none, only the given coords
        are interpolated (i.e. neighborhood = (0,0,0) ).
    device: torch device.
    """
    if neighborhood_points is not None:
        n_input_points = coords.shape[0]

        # Extend the coords array with the neighborhood coordinates
        coords = extend_coordinates_with_neighborhood(
            coords, neighborhood_points)

        # Interpolate signal for each (new) point
        coords_torch = torch.as_tensor(coords, dtype=torch.float,
                                       device=device)
        flat_subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                         coords_torch)

        # Reshape signal into (n_points, new_nb_features)
        # DWI data features for each neighbor are concatenated.
        #    dwi_feat1_neig1  dwi_feat2_neig1 ...  dwi_featn_neighbm
        #  p1        .              .                    .
        #  p2        .              .                    .
        n_features = (flat_subj_x_data.shape[-1] *
                      (neighborhood_points.shape[0] + 1))
        subj_x_data = flat_subj_x_data.reshape(n_input_points, n_features)

    else:  # No neighborhood:
        # Interpolate signal for each point
        coords_torch = torch.as_tensor(coords, dtype=torch.float,
                                       device=device)
        subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                    coords_torch)

    return subj_x_data, coords_torch
