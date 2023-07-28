# -*- coding: utf-8 -*-
import itertools
from typing import Union, List

import numpy as np
import torch


def prepare_neighborhood_vectors(
        neighborhood_type: str, neighborhood_radius: int,
        neighborhood_resolution: float = 1.0):
    """
    Prepare neighborhood vectors.

    Note: We only support isometric voxels! Adding isometry would also require
    the voxel resolution.

    Params
    ------
    neighborhood_type: str
        Either the 'axes' option or the 'grid' option. See each method for a
        description.
    neighborhood_radius: Optional[int]
        The radius. For the axes option: a radius of 1 = 7 neighbhors, a
        radius of 2 = 13 . For the grid option: a radius of 1 = 27 neighbors,
        a radius of 2 = 125.
    neighborhood_resolution: Optional[float]
        Resolution between each layer of neighborhood, in voxel space as compared
        to the MRI data. Ex: 0.5 one neighborhood every half voxel.
        Hint: Neighborhood's space will depend on this value. To convert from
        mm to voxel world, you may use
        dwi_ml.data.processing.space.world_to_vox.convert_world_to_vox(
            radius_mm, affine_mm_to_vox)

    Returns
    -------
    neighborhood_vectors: tensor of shape (N, 3).
        Results are vectors pointing to a neighborhood point, starting from the
        origin (i.e. current position). The current point (0,0,0) is included.
        Hint: You can now interpolate your DWI data in each direction around
        your point of interest to get your neighbourhood.
        Returns None if neighborhood_radius is None.
    """
    if neighborhood_type is None:
        return None

    if neighborhood_radius is None:
        raise ValueError("You must provide neighborhood radius to add "
                         "a neighborhood.")
    if neighborhood_resolution is None:
        raise ValueError("You must provide neighborhood resolution to add "
                         "a neighborhood.")
    if neighborhood_type not in ['axes', 'grid']:
        raise ValueError(
            "Neighborhood type must be either 'axes', 'grid' "
            "but we received {}!".format(neighborhood_type))

    if neighborhood_type == 'axes':
        neighborhood_vectors = get_neighborhood_vectors_axes(
            neighborhood_radius, neighborhood_resolution)
    else:
        neighborhood_vectors = get_neighborhood_vectors_grid(
            neighborhood_radius, neighborhood_resolution)

    neighborhood_vectors = torch.cat((torch.zeros(1, 3),
                                      neighborhood_vectors))
    return neighborhood_vectors


def get_neighborhood_vectors_axes(radius: int, resolution: float):
    """
    This neighborhood definition lies on a sphere. Returns a list of 7
    positions (current, up, down, left, right, behind, in front) at exactly
    `resolution` (mm or voxels) from origin (i.e. current postion).
    If radius is > 1, returns a multi-radius neighborhood (lying on
    concentring spheres).

    Returns
    -------
    neighborhood_vectors : tensor of shape (N, 3)
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). The current point (0,0,0) is
        included.
    """
    tmp_axes = np.identity(3)
    unit_axes = np.concatenate((tmp_axes, -tmp_axes))

    radiuses = np.asarray(range(radius)) * resolution

    neighborhood_vectors = []
    for r in radiuses:
        neighborhood_vectors.extend(unit_axes * r)
    neighborhood_vectors = torch.as_tensor(np.asarray(neighborhood_vectors),
                                           dtype=torch.float)

    return neighborhood_vectors


def get_neighborhood_vectors_grid(radius: int, resolution: float):
    """
    This neighborhood definition lies on a grid. Returns a list of vectors
    pointing to points surrounding the origin that mimic the original voxel
    grid, in voxel space. Ex: with radius 1, this is 27 points. With radius 2,
    it's 125 points.

    Returns
    -------
    neighborhood_vectors : tensor shape (N, 3)
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). The current point (0,0,0) is NOT
        included. Final neighboorhood will be of dimension 2*(radius+1) ^3.
    """
    neighborhood_vectors = []
    the_range = range(-radius, radius + 1)
    for x, y, z in itertools.product(the_range, the_range, the_range):
        if not (x == y == z == 0):  # Not adding origin; not a neighbor
            neighborhood_vectors.append([x, y, z])

    neighborhood_vectors = np.asarray(neighborhood_vectors) * resolution
    neighborhood_vectors = torch.as_tensor(neighborhood_vectors,
                                           dtype=torch.float)

    return neighborhood_vectors


def extend_coordinates_with_neighborhood(
        coords: torch.Tensor, neighborhood_vectors: torch.tensor):
    """
    From a list of coordinates and neighborhood vectors (e.g. [up, down, left,
    right]), get a new list of coordinates with all translations applied to all
    coordinates.

    Parameters
    ------
    coords: tensor of shape (M, 3)
        An array of M points; [x,y,z] coordinates.
    neighborhood_vectors: tensor of shape (N, 3)
        A list of translation vectors to apply to each point in coords.

    Returns
    -------
    flat_coords: tensor of shape (M x (N+1), 3)
        The new coordinates with all N neighbors (N+1 including the original
        coordinates), in the same space and origin as coords.
    tiled_vectors: tensor
        The coordinates of neighbors per respect to the current coordinate
        (translation vectors).
    """
    device = neighborhood_vectors.device
    assert coords.device == device, "Neighborhood device is {}, but current " \
                                    "coordinates device is {}" \
                                    .format(device, coords.device)

    m_coords = coords.shape[0]
    n_neighbors = neighborhood_vectors.shape[0]

    # 1. We repeat each coordinate to have the neighborhood size before
    # applying translations.
    # coords = [p1 p1... p2 p2 ... ...]'  (Size = [n, 3])
    flat_coords = coords.repeat_interleave(n_neighbors, dim=0)

    # 2. We translate each point based on the translations vector.
    # Ex, if neighborhood_translations = [here, up, down, left, right, ...]
    # coords = [p1+0 p1+up p1+down ..., p2+0 p2+up, p2+down, ...]'
    tiled_vectors = torch.tile(neighborhood_vectors, (m_coords, 1))
    flat_coords += tiled_vectors

    return flat_coords, tiled_vectors


def _unflatten_neighborhood_tensor(data_in_neighb: torch.Tensor, neighb):
    # When we do our interpolation, each tensor is one neighborhood per
    # coordinate.
    nb_points = data_in_neighb.shape[0]
    nb_neighb = len(neighb)

    # 1. Verifying that we have a grid neighborhood.
    # Rounding because, np.cbrt(27) = 3.0000000000000004
    out_size = np.round(np.cbrt(nb_neighb), decimals=6)
    assert out_size % 2 == 1, "Expecting a grid neighborhood to be able to " \
                              "unflatten. Not sure it is the case here! \n" \
                              "Number of neighbors should be 1, 27, 125, etc. " \
                              "but got {}".format(out_size)
    out_size = int(out_size)

    # 2. Finding the number of features.
    # Each point a list of len nb_neighbors * nb_features
    assert data_in_neighb.shape[1] % nb_neighb == 0, \
        "Data to be unflattend does not contain a multiple of expected " \
        "neighborhood size."
    nb_features = int(data_in_neighb.shape[1] / nb_neighb)

    # 3. Grid is equally sampled, with a 'voxel size' defined by user.
    raise NotImplementedError

    # 4.
    # The way we perform our interpolation, we get:
    # n1 - f1, n1 - f2, ....,  n2 - f1, n2 - f2, ...
    unflattened = \
        torch.zeros((nb_points, out_size, out_size, out_size, nb_features))
    for n, neibh_n in enumerate(neighb):
        print("NEIGH ", neibh_n)
        print("GETTING", data_in_neighb[:, n*nb_features:(n+1)*nb_features])
        print("INSIDE", unflattened[:, neibh_n[0], neibh_n[1], neibh_n[2], :])
        unflattened[:, neibh_n[0], neibh_n[1], neibh_n[2], :] = \
            data_in_neighb[:, n*nb_features:(n+1)*nb_features]

    print(data_in_neighb)
    print(data_in_neighb.shape)
    raise NotImplementedError


def unflatten_neighborhood(
        data_in_neighb: Union[torch.Tensor, List[torch.Tensor]], neighb):
    if isinstance(data_in_neighb, list):
        return [_unflatten_neighborhood_tensor(d, neighb) for d in
                data_in_neighb]
    else:
        return _unflatten_neighborhood_tensor(data_in_neighb, neighb)
