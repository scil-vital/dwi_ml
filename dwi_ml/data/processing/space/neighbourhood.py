# -*- coding: utf-8 -*-
import itertools
from typing import Iterable, Union

import numpy as np


def get_neighborhood_vectors_axes(radius: Union[float, Iterable[float]]):
    """This neighborhood definition lies on a sphere. Returns a list of 6
    positions (up, down, left, right, behind, in front) at exactly `radius`
    length. Good for RNN, for example.
    If radius is an iterable of floats, returns a multi-radius neighborhood.

    Hint: If you know your radius in mm only, use
    dwi_ml.data.processing.space.convert_world_to_vox.convert_world_to_vox
    Ex: radius_vox = convert_mm2vox(radius_mm, affine_mm_to_vox)

    Note: We only support isometric voxels! Adding isometry would also require
    remembering where the x,y,z directions are.

    Parameters
    ----------
    radius : number (int or float) or list or np.ndarray.
        Distance to each neighbor on a sphere (in voxel space).

    Returns
    -------
    neighborhood : np.ndarray[float]
        A list of vectors with last dimension = 3 (x,y,z coordinate for each
        neighbour per respect to the origin). Hint: You can now interpolate your
        DWI data in each direction around your point of interest to get your
        neighbourhood.
    """
    tmp_axes = np.identity(3)
    unit_axes = np.concatenate((tmp_axes, -tmp_axes))

    if not isinstance(radius, Iterable):
        radius = [radius]

    neighborhood = []
    for r in radius:
        neighborhood.append(unit_axes * r)
    neighborhood = np.asarray(neighborhood)

    return neighborhood


def get_neighborhood_vectors_grid(radius: int):
    """Returns a list of points similar to the original voxel grid. Ex: with
    radius 1, this is 27 points. With radius 2, that's 125 points. Good for
    CNN, for example.

    Note: We only support isometric voxels! Adding isometry would also require
    remembering where the x,y,z directions are.

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
        your neighbourhood.
    """
    assert type(radius) == int

    neighborhood = []
    the_range = range(-radius, radius + 1)
    for x, y, z in itertools.product(the_range, the_range, the_range):
        neighborhood.append([x, y, z])
    neighborhood = np.asarray(neighborhood)

    return neighborhood


def extend_coordinates_with_neighborhood(coords: np.ndarray,
                                         neighborhood_translations: np.ndarray):
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
    tiled_vectors = np.tile(np.concatenate((np.zeros((1, 3)),
                                            neighborhood_translations)),
                            (n_coords, 1))
    coords += tiled_vectors

    return coords
