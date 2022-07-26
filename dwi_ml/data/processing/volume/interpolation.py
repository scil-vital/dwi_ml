# -*- coding: utf-8 -*-
import torch
import numpy as np

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


def torch_trilinear_interpolation(volume: torch.Tensor,
                                  coords_vox_corner: torch.Tensor):
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
    coords_vox_corner : torch.Tensor with shape (N,3)
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
    assert volume.device == coords_vox_corner.device,\
        "volume on device: {}; " \
        "coords on device: {}".format(volume.device, coords_vox_corner.device)
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
        torch.floor(coords_vox_corner[:, None, :] + idx_box_torch
                    ).reshape((-1, 3)).long()

    # Clip indices to make sure we don't go out-of-bounds
    # Origin = corner means the minimum is 0.
    #                       the maximum is shape.
    # Ex, for shape 150, last voxel is #149, with possible coords up to 149.99.
    lower = torch.as_tensor([0, 0, 0], device=device)
    upper = torch.as_tensor(volume.shape[:3], device=device) - 1
    idx_box_clipped = torch.min(torch.max(idx_box_unclipped, lower), upper)

    # Setting Q1 such as in equation 9.9
    d = coords_vox_corner - torch.floor(coords_vox_corner)
    dx, dy, dz = d[:, 0], d[:, 1], d[:, 2]
    Q1 = torch.stack([torch.ones_like(dx), dx, dy, dz,
                      dx * dy, dy * dz, dx * dz,
                      dx * dy * dz], dim=0)

    if volume.dim() == 3:
        # Fetch volume data at indices based on equation 9.11.
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2]]
        # Last dim (-1) = the 8 corners
        p = p.reshape((coords_vox_corner.shape[0], -1)).t()

        # Finding coordinates with equation 9.12a.
        output = torch.sum(p * torch.mm(B1_torch.t(), Q1), dim=0)

        return output
    elif volume.dim() == 4:

        # Fetch volume data at indices
        p = volume[idx_box_clipped[:, 0],
                   idx_box_clipped[:, 1],
                   idx_box_clipped[:, 2], :]
        p = p.reshape((coords_vox_corner.shape[0], 8, volume.shape[-1]))

        output = \
            torch.sum(p * torch.mm(B1_torch.t(), Q1).t()[:, :, None], dim=1)

        return output
    else:
        raise ValueError("Interpolation: There was a problem with the "
                         "volume's number of dimensions!")


def interpolate_volume_in_neighborhood(
        volume_as_tensor, coords_vox_corner, neighborhood_vectors_vox=None,
        add_vectors_to_data: bool = False, device=torch.device('cpu')):
    """
    Params
    ------
    data_tensor: tensor
        The data: a 4D tensor with last dimension F (nb of features).
    coords_vox_corner: np.array with shape (M, 3)
        A list of points (3d coordinates). Neighborhood will be added to these
        points based. Coords must be in voxel world, origin='corner', to use
        trilinear interpolation.
    neighborhood_vectors_vox: np.ndarray[float] with shape (N, 3)
        The neighboors to add to each coord. Do not include the current point
        ([0,0,0]). Values are considered in the same space as
        coords_vox_corner, and should thus be in voxel space.
    add_vectors_to_data: bool
        If true, neighborhood vectors will be concatenated to data at each
        point, meaning that the number of features per point will be 3 more.
    device: torch device.

    Returns
    -------
    subj_x_data: tensor of shape (M, F * (N+1))
        The interpolated data: M points with contatenated neighbors.
    flat_coords_torch: tensor of shape (M x (N+1), 3)
        The final coordinates.
    """
    if (neighborhood_vectors_vox is not None and
            len(neighborhood_vectors_vox) > 0):
        m_input_points = coords_vox_corner.shape[0]
        n_neighb = neighborhood_vectors_vox.shape[0]
        f_features = volume_as_tensor.shape[-1]

        # Extend the coords array with the neighborhood coordinates
        # coords: shape (M x (N+1), 3)
        flat_coords_with_neighb, tiled_vectors = \
            extend_coordinates_with_neighborhood(coords_vox_corner,
                                                 neighborhood_vectors_vox)

        flat_coords_torch = torch.as_tensor(flat_coords_with_neighb,
                                            dtype=torch.float, device=device)

        # Interpolate signal for each (new) point
        # DWI data features for each neighbor are concatenated.
        # Result is of shape: (M * (N+1), F).
        flat_subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                         flat_coords_torch)

        if add_vectors_to_data:
            # Concat (M * (N+1), F) with (M x (N+1), 3)
            flat_subj_x_data = torch.cat(
                (flat_subj_x_data, torch.tensor(tiled_vectors, device=device)),
                dim=-1)
            f_features += 3

        # Neighbors become new features of the current point.
        # Reshape signal into (M, (N+1)*F))
        new_nb_features = (f_features * (n_neighb + 1))
        subj_x_data = flat_subj_x_data.reshape(m_input_points, new_nb_features)

    else:  # No neighborhood:
        if add_vectors_to_data:
            raise ValueError("You should not select 'add_coordinates_to_data' "
                             "if you do not add a neighborhood; you would "
                             "only add a bunch of zeros...")
        
        # Interpolate signal for each point
        flat_coords_torch = torch.as_tensor(coords_vox_corner,
                                            dtype=torch.float, device=device)
        subj_x_data = torch_trilinear_interpolation(volume_as_tensor,
                                                    flat_coords_torch)

    return subj_x_data, flat_coords_torch
