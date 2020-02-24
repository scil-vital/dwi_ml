#!/usr/bin/env python

import itertools

import numpy as np


# checked!
def get_neighborhood_vectors_axes(radius):
    """
    This neighborhood definition lies on a sphere. Returns a list of 6 positions
    (up, down, left, right, behind, in front) at exactly `radius` length
    (+ 0,0,0, the origin). Good for RNN, for example.
    If radius is a list of floats, returns a multi-radius neighborhood.

    Hint: If you know your radius in mm only, use
    dwi_ml.data.processing.space.convert_world_to_vox.convert_world_to_vox
    Ex: radius_vox = convert_mm2vox(radius_mm, affine_mm_to_vox)

    Note: We only support isometric voxels! Adding isometry would also require
    remembering where the x,y,z directions are.

    Parameters
    ----------
    radius : number (int or float) or list or np.ndarray.
        Distance to each neighbor sphere (in voxel space).

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

    if type(radius) == list or type(radius) == np.ndarray:
        neighborhood = [[0, 0, 0]]
        for r in radius:
            neighborhood = np.concatenate((unit_axes*r, neighborhood))
    else:
        neighborhood = unit_axes * radius
        neighborhood = np.concatenate(([[0, 0, 0]], neighborhood))

    return neighborhood


# checked!
def get_neighborhood_vectors_grid(radius: int):
    """Returns a list of points similar to the original voxel grid. Ex: with
    radius 1, this is 27 points. With radius 2, that's 125 points. Good for CNN,
    for example.

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
        neighbour per respect to the origin). Hint: You can now interpolate your
        DWI data in each direction around your point of interest to get your
        neighbourhood.
    """
    assert type(radius) == int

    neighborhood = []
    the_range = range(-radius, radius + 1)
    for x, y, z in itertools.product(the_range, the_range, the_range):
        neighborhood.extend([[x, y, z]])
    neighborhood = np.asarray(neighborhood)

    return neighborhood


def extend_with_interp_neighborhood(list_points: np.ndarray,
                                    neighborhood_vectors: np.ndarray):
    """
    Get the interpolated neighborhood coordinates for each point in a list of
    (x,y,z) points.

    Params
    ------
    list_points: np.ndarray
        A list of points [point1 point2 ...] where each point is a [x y z]
        coordinate.
    neighborhood_vectors: np.ndarray
        A list of neighboors to add to each point in list_points.

    Returns
    -------
        list_point: a modified list of points, where ?
    """
    n_input_points = list_points.shape[0]

    # At the beginning
    # list_points = [[p1_x p1_y p1_z]
    #                [p2_x p2_y p2_z]]
    #             = [ p1
    #                 p2 ]
    #             = [ p1 p2 ]'

    # 1. We repeat each point to have the neighborhood size.
    # list_point = [p1 p1... p2 p2 ... ...]'
    list_points = np.repeat(list_points, neighborhood_vectors.shape[0], axis=0)

    # 2. We translate each point based on the neighborhood vector.
    # Ex, if neighborhood_vectors = [here, up, down, left, right, ...]
    # list_point = [p1+0 p1+up p1+down ..., p2+0 p2+up, p2+down, ...]'
    tiled_vectors = np.tile(neighborhood_vectors, (n_input_points, 1))
    list_points += tiled_vectors

    return list_points
