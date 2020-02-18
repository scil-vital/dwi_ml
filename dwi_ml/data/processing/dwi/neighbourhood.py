
from scilpy import convert_mm2vox # Fait partie des trucs qu'on veut envoyer sur scilpy. À voir.

import numpy as np
import itertools


# If you know your radius in mm only, use convert_mm_to_vox (should be sent to
# scilpy from dwi_ml soon). Ex:
# radius_vox = convert_mm2vox(radius_mm, affine_mm_to_vox)
def get_interp_neighborhood_vectors(radius_vox, method: str = "6axes") \
        -> np.ndarray:
    """
    Returns a neighbourhood based on the method choice.

    Parameters
    ----------
    radius_vox : float or list of floats.
        Distance to neighbors (in voxel space. Ex: 2 neighbours distance).
        See the "6axes" method for the list of floats usage. With the "grid"
        method, only a single float is accepted.
    method: str
        Choice of neigbhoorhood definition. Choices:
        "6axes": Returns a list of 6+1 positions. Up, down, left, right, behind,
                 in front at exactly `radius` length (+ 0,0,0).
                 If radius is a list of floats, returns a multi-radius
                 neighborhood, with 6 positions per radius (+ 0,0,0).
                 Good for RNN, for example.
        "grid":  Returns a 3D volume similar to the original voxel grid,
                 around (0,0,0,). Dimension 2*radius x 2*radius x 2*radius.
                 Good for CNN, for example.

    Returns
    -------
    directions : np.ndarray with last dimension = 3 (x,y,z coordinate for each
                 neighbour).
                 Coordinates are in voxel-space around the (0,0,0) coordinate.
                 You will need to interpolate the dwi in each direction around
                 your point of interest to get your neighbourhood.
    """
    # toDo: This only works with iso voxels! We might have to change this one day
    #  with radius_vox_x, radius_vox_y, radius_vox_z. But we need to check how
    #  to find where the x dimension is.
    #
    # toDo: add method "n_axes"

    if method == "6axes":
        unit_axes = _get_6_unit_axes()

        if type(radius_vox) == float:
            neighborhood = unit_axes * radius_vox
            neighborhood = np.concatenate(([[0, 0, 0]], neighborhood))
        else:
            neighborhood = [[0, 0, 0]]
            for r in radius_vox:
                neighborhood = np.concatenate((unit_axes*r, neighborhood))
    elif method == "grid":
        if type(radius_vox) == float:
            neighborhood = []
            the_range = range(-radius_vox, radius_vox+1)
            for x, y, z in itertools.product(the_range, the_range, the_range):
                neighborhood.extend([[x, y, z]])
            neighborhood = np.asarray(neighborhood)
        else:
            raise ValueError("You can't use a list of floats as radius_vox arg"
                             "together with the grid method.")
    else:
        raise NotImplementedError

    return neighborhood


def _get_6_unit_axes():
    tmp_axes = np.identity(3)
    unit_axes = np.concatenate((tmp_axes, -tmp_axes))

    return unit_axes


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
