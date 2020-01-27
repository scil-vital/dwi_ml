"""Streamline transformation utilities. All functions should take a stateful
tractogram as input and manage the streamline space.

Functions:
    remove_short_streamlines_in_sft
    resample_sft
    subsample_sft
    compress_sft
"""

import logging

import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.segment.clustering import QuickBundlesX
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamlinespeed import (
    compress_streamlines, length, set_number_of_points)
from scil_vital.shared.code.transformation.streamlines import \
    remove_similar_streamlines


def remove_short_streamlines_from_sft(tractogram: StatefulTractogram,
                                      min_length_mm: float):
    # When ready for python 3.7: -> StatefulTractogram:
    """Remove all streamlines shorter than the minimum length in mm.

    Parameters
    ----------
    tractogram : StatefulTractogram
        Tractogram to filter.
    min_length_mm : float
        Streamlines shorter than this length will be removed.

    Returns
    -------
    tractogram : StatefulTractogram
        A tractogram without short streamlines.
    """
    # Make sure we are in world space
    orig_space = tractogram.space
    tractogram.to_rasmm()

    lengths = length(tractogram.streamlines)
    filtered_streamlines = [s for (s, l) in zip(tractogram.streamlines, lengths)
                            if l > min_length_mm]
    output_tractogram = StatefulTractogram(
        filtered_streamlines, tractogram, Space.RASMM,
        shifted_origin=tractogram.shifted_origin)

    # Return to original space
    if orig_space == Space.VOX:
        output_tractogram.to_vox()
    elif orig_space == Space.VOXMM:
        output_tractogram.to_voxmm()

    return output_tractogram


def resample_sft(tractogram: StatefulTractogram,
                 step_size: float) -> StatefulTractogram:
    """Resample streamlines to have a constant step size, from a stateful
    tractogram. See also: resample_streamlines_step_size.

    Parameters
    ----------
    tractogram : dipy.io.stateful_tractogram.StatefulTractogram
        Tractogram to resample.
    step_size : float
        Step size that all streamlines should have.
    Returns
    -------
    output_tractogram : dipy.io.stateful_tractogram.StatefulTractogram
        Tractogram with all streamlines resampled with a constant step size.
    """
    # Make sure we are in world space
    orig_space = tractogram.space
    tractogram.to_rasmm()

    # Resample streamlines
    lengths = length(tractogram.streamlines)
    nb_points = np.ceil(lengths / step_size).astype(int)
    if np.any(nb_points == 1):
        # Some streamlines are too short;
        # make sure there is always at least 2 points
        logging.warning("Some streamlines are shorter than the provided "
                        "step size...")
        nb_points[nb_points == 1] = 2
    streamlines_resampled = [set_number_of_points(s, n) for s, n in
                             zip(tractogram.streamlines, nb_points)]
    output_tractogram = StatefulTractogram(streamlines_resampled, tractogram,
                                           Space.RASMM,
                                           tractogram.shifted_origin)
    # Return to original space
    if orig_space == Space.VOX:
        tractogram.to_vox()
    elif orig_space == Space.VOXMM:
        tractogram.to_voxmm()

    return output_tractogram


def subsample_sft(tractogram: StatefulTractogram,
                  clustering_threshold_mm: float,
                  removal_distance_mm: float):
    # When ready for python 3.7: -> StatefulTractogram:
    """Subsample a group of streamlines using a distance-based similarity.

    Streamlines are first clustered using `clustering_threshold_mm`, then, for
    each cluster, similar streamlines (with an mdf distance smaller than
    `removal_distance_mm`) are reduced to a single streamline.

    Parameters
    ----------
    tractogram : StatefulTractogram
        Streamlines to subsample.
    clustering_threshold_mm : float
        Distance threshold to group streamlines into smaller clusters.
    removal_distance_mm : float
        Threshold to reduce similar streamlines to a single streamline.

    Returns
    -------
    output_tractogram : StatefulTractogram
        Subsampled streamlines
    """

    output_streamlines = []

    # Bring streamlines to RASmm space
    orig_space = tractogram.space
    tractogram.to_rasmm()

    feature = ResampleFeature(nb_points=20)
    metric = AveragePointwiseEuclideanMetric(feature)

    # Use a 2-level clustering strategy
    qb = QuickBundlesX(thresholds=[2 * clustering_threshold_mm,
                                   clustering_threshold_mm], metric=metric)
    cluster_tree = qb.cluster(tractogram.streamlines)

    # Subsample streamlines in each cluster
    for cluster in cluster_tree.get_clusters(2):
        streamlines = [tractogram.streamlines[i] for i in cluster]

        # Remove similar streamlines
        subsample = remove_similar_streamlines(streamlines, removal_distance_mm)
        output_streamlines.extend(subsample)

    output_tractogram = StatefulTractogram(output_streamlines, tractogram,
                                           Space.RASMM,
                                           tractogram.shifted_origin)

    # Return to original space
    if orig_space == Space.VOX:
        output_tractogram.to_vox()
    elif orig_space == Space.VOXMM:
        output_tractogram.to_voxmm()

    return output_tractogram


def compress_sft(tractogram: StatefulTractogram):
    # When ready for python 3.7: -> StatefulTractogram:
    """Compress the streamlines of a stateful tractogram

    Parameters
    ----------
    tractogram : StatefulTractogram

    Returns
    -------
    output_tractogram : StatefulTractogram
    """
    orig_space = tractogram.space
    tractogram.to_rasmm()
    compressed_streamlines = compress_streamlines(tractogram.streamlines)

    output_tractogram = StatefulTractogram(compressed_streamlines, tractogram,
                                           tractogram.space,
                                           tractogram.shifted_origin)

    if orig_space == Space.VOXMM:
        output_tractogram.to_voxmm()
    elif orig_space == Space.VOX:
        output_tractogram.to_vox()
    elif orig_space == Space.RASMM:
        pass
    else:
        raise ValueError

    return output_tractogram
