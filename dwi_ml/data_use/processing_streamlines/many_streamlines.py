"""Streamline transformation utilities. All functions should take a list of
streamlines or an iterable as input and manage the streamline space.

Functions:
    remove_similar_streamlines
    resample_streamlines
    apply_transform_to_streamlines
    cut_random_streamlines
"""

from __future__ import annotations

from typing import Iterable, List, Union

import nibabel as nib
import numpy as np
from dipy.align.bundlemin import distance_matrix_mdf
from dipy.tracking.streamlinespeed import length, set_number_of_points
from scipy.stats import truncnorm
from scil_vital.shared.code.transformation.space import \
    convert_mm2vox
from scil_vital.shared.code.transformation.streamline import \
    split_array_at_lengths

                                                                                                            #toDo: the type for streamlines is never the same!!


def remove_similar_streamlines(streamlines: List[np.ndarray],
                               removal_distance: float):
    """Computes a distance matrix using all streamlines,
    then removes streamlines closer than `removal_distance`.


    Parameters
    ----------
    streamlines : list of np.ndarray
        Streamlines to downsample.
    removal_distance : float
        Distance for which streamlines are considered 'similar' and should be
        removed (in the space of the tracks).

    Returns
    -------
    streamlines : list of `numpy.ndarray` objects
        Downsampled streamlines
    """
    if len(streamlines) <= 1:
        return streamlines

    # Simple trick to make it faster than using 20 points
    sample_10_streamlines = set_number_of_points(streamlines, 10)
    distance_matrix = distance_matrix_mdf(sample_10_streamlines,
                                          sample_10_streamlines)

    current_id = 0
    while True:
        indices = np.where(distance_matrix[current_id] < removal_distance)[0]

        it = 0
        if len(indices) > 1:
            for k in indices:
                # Every streamlines similar to yourself (excluding yourself)
                # should be deleted from the set of desired streamlines
                if not current_id == k:
                    streamlines.pop(k - it)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=0)
                    distance_matrix = np.delete(distance_matrix, k - it, axis=1)
                    it += 1

        current_id += 1
        # Once you reach the end of the remaining streamlines
        if current_id >= len(streamlines):
            break

    return streamlines


def apply_transform_to_streamlines(streamlines: Iterable, affine: np.ndarray)\
        -> nib.streamlines.ArraySequence:
    """Apply an affine transformation on a set of streamlines

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence or List of np.ndarray
        Streamlines to transform.
    affine : np.ndarray with shape (4,4)
        Affine tranformation to apply on the streamlines.

    Returns
    -------
    nib.streamlines.ArraySequence
        Transformed streamlines.
    """
    tractogram = nib.streamlines.Tractogram(streamlines)
    tractogram.apply_affine(affine)
    return tractogram.streamlines


