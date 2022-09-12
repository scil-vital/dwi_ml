# -*- coding: utf-8 -*-
import itertools
import logging
from typing import Iterable, Union

import numpy as np


def prepare_neighborhood_vectors(neighborhood_type, neighborhood_radius):
    """
    Prepare neighborhood vectors either with the 'axes' option or the
    'grid' option. See each method for a description.

    Results are in the voxel world: vectors pointing to a neighborhood point,
    starting at the origin (i.e. current position).

    The current point (0,0,0) is NOT included.
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
            neighborhood_points = get_neighborhood_vectors_axes(
                neighborhood_radius)
        else:
            neighborhood_points = get_neighborhood_vectors_grid(
                neighborhood_radius)
        return neighborhood_points
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
    voxel from origin (i.e. current postion). If radius is an iterable of
    floats, returns a multi-radius neighborhood (lying on concentring spheres).

    Hint: If you know your radius in mm only, use
    dwi_ml.data.processing.space.world_to_vox.convert_world_to_vox
    Ex: radius_vox = convert_world_to_vox(radius_mm, affine_mm_to_vox)

    Note: We only support isometric voxels! Adding isometry would also require
    the voxel resolution.

    Parameters
    ----------
    radius : number (int or float) or iterable of numbers.
        Distance to each neighbor (in voxel space).

    Returns
    -------
    neighborhood : np.ndarray[float]
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). Hint: You can now interpolate
        your DWI data in each direction around your point of interest to get
        your neighbourhood. The current point (0,0,0) is NOT included.
    """
    tmp_axes = np.identity(3)
    unit_axes = np.concatenate((tmp_axes, -tmp_axes))

    if not isinstance(radius, Iterable):
        radius = [radius]

    neighborhood = []
    for r in radius:
        neighborhood.extend(unit_axes * r)
    neighborhood = np.asarray(neighborhood)

    return neighborhood


def get_neighborhood_vectors_grid(radius: int):
    """
    This neighborhood definition lies on a grid. Returns a list of vectors
    pointing to points surrounding the origin that mimic the original voxel
    grid, in voxel space. Ex: with radius 1, this is 26 points. With radius 2,
    it's 124 points.

    Note: We only support isometric voxels! Adding anisometry would also
    require remembering the voxel resolution.

    Parameters
    ----------
    radius : int
        Size of the neighborhood in each direction, in voxel space. Final
        neighboorhood will be of dimension 2*radius x 2*radius x 2*radius.

    Returns
    -------
    neighborhood : np.ndarray[float]
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). Hint: You can now interpolate
        your DWI data in each direction around your point of interest to get
        your neighbourhood. The current point (0,0,0) is NOT included.
    """
    assert type(radius) == int

    neighborhood = []
    the_range = range(-radius, radius + 1)
    for x, y, z in itertools.product(the_range, the_range, the_range):
        if not (x == y == z == 0):  # Not adding origin; not a neighbor
            neighborhood.append([x, y, z])
    neighborhood = np.asarray(neighborhood)

    return neighborhood


def extend_coordinates_with_neighborhood(
        coords: np.ndarray, neighborhood_translations: np.ndarray):
    """
    From a list of coordinates (e.g. [p1,p2,p3]) and neighborhood translation
    vectors (e.g. [up, down, left, right]), get a new list of coordinates
    with the product of all translations applied to all coordinates (new length
    will be `n_coords + n_coords x n_translations`)

    Parameters
    ------
    coords: np.ndarray with shape (N, 3)
        An array of [x,y,z] coordinates [p1, p2, ...].
    neighborhood_translations: np.ndarray with shape (M, 3)
        A list of translation vectors to apply to each point in coords.

    Returns
    -------
    coords: np.ndarray with shape (N x (M+1), 3)
        The new coordinates with all translations applied to all
        coordinates, including the original coordinates.
    """
    n_coords = coords.shape[0]
    n_neighbors = neighborhood_translations.shape[0]

    # 1. We repeat each coordinate to have the neighborhood size (+ 1 for
    # original coordinate) before applying translations.
    # coords = [p1 p1... p2 p2 ... ...]'
    coords = np.repeat(coords, n_neighbors + 1, axis=0)

    # 2. We translate each point based on the translations vector.
    # Ex, if neighborhood_translations = [here, up, down, left, right, ...]
    # coords = [p1+0 p1+up p1+down ..., p2+0 p2+up, p2+down, ...]'
    total_neighborhood = np.concatenate((np.zeros((1, 3)),
                                         neighborhood_translations))
    tiled_vectors = np.tile(total_neighborhood, (n_coords, 1))
    coords += tiled_vectors

    return coords
