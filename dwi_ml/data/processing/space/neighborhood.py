# -*- coding: utf-8 -*-
import itertools
import logging
from typing import Iterable, Union

import numpy as np


def prepare_neighborhood_vectors(neighborhood_type: str, neighborhood_radius):
    """
    Prepare neighborhood vectors.

    Params
    ------
    neighborhood_type: str
        Either the 'axes' option or the 'grid' option. See each method for a
        description.
    neighborhood_radius: Union[int, float, Iterable[float], None]
        The radius. A int for the grid option. An float or list of floats for
        the axes option. With the grid option, radius must be given in voxel
        space.

    Returns
    -------
    neighborhood_vectors: np.ndarray[float] with shape (N, 3).
        Results are vectors pointing to a neighborhood point, starting from the
        origin (i.e. current position). The current point (0,0,0) is NOT
        included.
        Hint: You can now interpolate your DWI data in each direction around
        your point of interest to get your neighbourhood.
        Returns None if neighborhood_radius is None.
    """
    if neighborhood_type is not None:
        if neighborhood_radius is None:
            raise ValueError("You must provide neighborhood radius to add "
                             "a neighborhood.")

        if neighborhood_type not in ['axes', 'grid']:
            raise ValueError(
                "Neighborhood type must be either 'axes', 'grid' "
                "but we received {}!".format(neighborhood_type))

        if neighborhood_type == 'axes':
            neighborhood_vectors = get_neighborhood_vectors_axes(
                neighborhood_radius)
        else:
            if isinstance(neighborhood_radius, list):
                assert len(neighborhood_radius) == 1
                neighborhood_radius = neighborhood_radius[0]
            neighborhood_vectors = get_neighborhood_vectors_grid(
                neighborhood_radius)

        return neighborhood_vectors
    else:
        if neighborhood_radius is not None:
            logging.debug(
                "You have chosen not to add a neighborhood (value "
                "None), but you have given a neighborhood radius. "
                "Discarded.")
        return None


def get_neighborhood_vectors_axes(radius: Union[float, Iterable[float]]):
    """
    This neighborhood definition lies on a sphere. Returns a list of 6
    positions (up, down, left, right, behind, in front) at exactly `radius`
    (mm or voxels) from origin (i.e. current postion). If radius is an iterable
    of floats, returns a multi-radius neighborhood (lying on concentring
    spheres).

    Hint: Neighborhood's space will depend on the radius you give. To convert,
    from mm to voxel world, you may use
    dwi_ml.data.processing.space.world_to_vox.convert_world_to_vox(
                                                   radius_mm, affine_mm_to_vox)

    Note: We only support isometric voxels! Adding isometry would also require
    the voxel resolution.

    Parameters
    ----------
    radius : number (int or float) or iterable of numbers.
        Distance to each neighbor.

    Returns
    -------
    neighborhood_vectors : np.ndarray[float] with shape (N, 3)
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). The current point (0,0,0) is NOT
        included.
    """
    tmp_axes = np.identity(3)
    unit_axes = np.concatenate((tmp_axes, -tmp_axes))

    if not isinstance(radius, Iterable):
        radius = [radius]

    neighborhood_vectors = []
    for r in radius:
        neighborhood_vectors.extend(unit_axes * r)
    neighborhood_vectors = np.asarray(neighborhood_vectors)

    return neighborhood_vectors


def get_neighborhood_vectors_grid(radius_vox_space: int):
    """
    This neighborhood definition lies on a grid. Returns a list of vectors
    pointing to points surrounding the origin that mimic the original voxel
    grid, in voxel space. Ex: with radius 1, this is 26 points. With radius 2,
    it's 124 points.

    Note: We only support isometric voxels! Adding anisometry would also
    require remembering the voxel resolution.

    Parameters
    ----------
    radius_vox_space : int
        Size of the neighborhood in each direction, in voxel space. Final
        neighboorhood will be of dimension 2*radius x 2*radius x 2*radius.

    Returns
    -------
    neighborhood_vectors : np.ndarray[float] with shape (N, 3)
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). The current point (0,0,0) is NOT
        included.
    """
    if isinstance(radius_vox_space, float) and radius_vox_space.is_integer():
        radius_vox_space = int(radius_vox_space)
    assert type(radius_vox_space) == int, "For the 'grid' neighborhood, " \
                                          "radius must be an int. Rrecieved " \
                                          "{}".format(radius_vox_space)

    neighborhood_vectors = []
    the_range = range(-radius_vox_space, radius_vox_space + 1)
    for x, y, z in itertools.product(the_range, the_range, the_range):
        if not (x == y == z == 0):  # Not adding origin; not a neighbor
            neighborhood_vectors.append([x, y, z])
    neighborhood_vectors = np.asarray(neighborhood_vectors)

    return neighborhood_vectors


def extend_coordinates_with_neighborhood(
        coords: np.ndarray, neighborhood_vectors: np.ndarray):
    """
    From a list of coordinates and neighborhood vectors (e.g. [up, down, left,
    right]), get a new list of coordinates with all translations applied to all
    coordinates.

    Parameters
    ------
    coords: np.ndarray with shape (M, 3)
        An array of M points; [x,y,z] coordinates.
    neighborhood_vectors: np.ndarray[float] with shape (N, 3)
        A list of translation vectors to apply to each point in coords.

    Returns
    -------
    flat_coords: np.ndarray[float] with shape (M x (N+1), 3)
        The new coordinates with all N neighbors (N+1 including the original
        coordinates), in the same space and origin as coords.
    """
    m_coords = coords.shape[0]
    n_neighbors = neighborhood_vectors.shape[0]

    # 1. We repeat each coordinate to have the neighborhood size (+ 1 for
    # original coordinate) before applying translations.
    # coords = [p1 p1... p2 p2 ... ...]'
    flat_coords = np.repeat(coords, n_neighbors + 1, axis=0)

    # 2. We translate each point based on the translations vector.
    # Ex, if neighborhood_translations = [here, up, down, left, right, ...]
    # coords = [p1+0 p1+up p1+down ..., p2+0 p2+up, p2+down, ...]'
    total_neighborhood = np.concatenate((np.zeros((1, 3)),
                                         neighborhood_vectors))
    tiled_vectors = np.tile(total_neighborhood, (m_coords, 1))
    flat_coords += tiled_vectors

    return flat_coords, tiled_vectors
