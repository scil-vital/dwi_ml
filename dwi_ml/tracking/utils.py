# -*- coding: utf-8 -*-
from enum import Enum
from typing import Callable, Dict, Tuple

import nibabel as nib
import numpy as np
from dipy.tracking.utils import length as length_generator
from nibabel.affines import apply_affine

from dwi_ml.data.processing.volume.interpolation import (
    interpolate_volume_at_coordinates)


class StoppingFlags(Enum):
    """ Predefined stopping flags to use when checking which streamlines
    should stop """
    STOPPING_MASK = int('00000001', 2)
    STOPPING_LENGTH = int('00000010', 2)
    STOPPING_CURVATURE = int('00000100', 2)


def is_outside_mask(streamlines: np.ndarray, mask: np.ndarray,
                    affine_vox2mask: np.ndarray = None, threshold: float = 0.):
    """Checks which streamlines have their last coordinates outside a mask.

    Parameters
    ----------
    streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space.
    mask : 3D np.ndarray
        3D image defining a stopping mask. The interior of the mask is defined
        by values higher or equal than `threshold` .
        NOTE: The mask coordinates can be in a different space than the
        streamlines coordinates if an affine is provided.
    affine_vox2mask : np.ndarray with shape (4,4) (optional)
        Tranformation that aligns streamlines on top of `mask`.
    threshold : float
        Voxels with a value higher or equal than this threshold are considered
        as part of the interior of the mask.

    Returns
    -------
    outside : np.ndarray with shape (n_streamlines,)
        Bool array telling whether a streamline's last coordinate is outside
        the mask or not.
    """

    # Get last streamlines coordinates
    indices_vox = streamlines[:, -1, :]

    # Use affine to map coordinates in mask space
    indices_mask = nib.affines.apply_affine(affine_vox2mask, indices_vox)

    mask_values = interpolate_volume_at_coordinates(mask, indices_mask, mode='constant')
    outside = mask_values < threshold
    return outside


def is_too_long(streamlines: np.ndarray, max_nb_steps: int = None,
                max_length: float = None):
    """Checks whether streamlines have exceeded the maximum number of steps or a
    maximum length (not mutually exclusive).

    Parameters
    ----------
    streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space.
    max_nb_steps : int
        Maximum number of steps a streamline can have.
    max_length : float
        Maximum length a streamline can have.

    Returns
    -------
    too_long : np.ndarray with shape (n_streamlines,)
        Bool array telling whether a streamline is too long or not.
    """

    if not max_nb_steps and not max_length:
        raise ValueError("At least one of max_nb_steps or max_length must be "
                         "defined!")

    output = np.zeros(streamlines.shape[0])

    if max_nb_steps:
        output = np.logical_or(output,
                               np.asarray([streamlines.shape[1] >
                                           max_nb_steps] * len(streamlines)))

    if max_length:
        output = np.logical_or(output,
                               np.array(list(length_generator(streamlines)))
                               > max_length)

    return output


def is_too_curvy(streamlines: np.ndarray, max_theta: float):
    """Checks whether streamlines have exceeded the maximum angle between the
    last 2 steps

    Parameters
    ----------
    streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space.
    max_theta : float
        Maximum angle in degrees that two consecutive segments can have between
         each other.

    Returns
    -------
    too_curvy : np.ndarray with shape (n_streamlines,)
        Bool array telling whether a streamline is too curvy or not.
    """
    max_theta_rad = np.deg2rad(max_theta)  # Internally use radian

    if streamlines.shape[1] < 3:
        # Not enough segments to compute curvature
        return np.asarray([False] * len(streamlines))

    # Compute vectors for the last and before last streamline segments
    u = streamlines[:, -1] - streamlines[:, -2]
    v = streamlines[:, -2] - streamlines[:, -3]

    # Normalize vectors
    u /= np.sqrt(np.sum(u ** 2, axis=1, keepdims=True))
    v /= np.sqrt(np.sum(v ** 2, axis=1, keepdims=True))

    # Compute angles
    cos_theta = np.sum(u * v, axis=1).clip(-1., 1.)
    angles = np.arccos(cos_theta)

    return angles > max_theta_rad


def filter_stopping_streamlines(
        streamlines: np.ndarray,
        stopping_criteria: Dict[StoppingFlags, Callable]) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Checks which streamlines should stop and which ones should continue.

    Parameters
    ----------
    streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
        Streamline coordinates in voxel space.
    stopping_criteria : dict of int->Callable
        List of functions that take as input streamlines,
        and output a boolean numpy array indicating which streamlines should
        stop.

    Returns
    -------
    should_continue : np.ndarray
        Indices of the streamlines that should continue.
    should_stop : np.ndarray
        Indices of the streamlines that should stop.
    flags : np.ndarray
        `StoppingFlags` that triggered stopping for each stopping streamline.
    """
    idx = np.arange(len(streamlines))

    should_stop = np.zeros(len(idx), dtype=bool)
    flags = np.zeros(len(idx), dtype=np.uint8)

    # For each possible flag, determine which streamline should stop and keep
    # track of the triggered flag
    for flag, stopping_criterion in stopping_criteria.items():
        stopped_by_criterion = stopping_criterion(streamlines)
        should_stop[stopped_by_criterion] = True
        flags[stopped_by_criterion] |= flag.value

    should_continue = np.logical_not(should_stop)

    return idx[should_continue], idx[should_stop], flags[should_stop]


def is_flag_set(flags, ref_flag):
    """ Checks which flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value

    return ((flags.astype(np.uint8) & ref_flag) >>
            np.log2(ref_flag).astype(np.uint8)).astype(bool)


def count_flags(flags, ref_flag):
    """ Counts how many flags have the `ref_flag` set. """
    if type(ref_flag) is StoppingFlags:
        ref_flag = ref_flag.value

    return is_flag_set(flags, ref_flag).sum()
