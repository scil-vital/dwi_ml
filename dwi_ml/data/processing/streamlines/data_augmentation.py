# -*- coding: utf-8 -*-

from collections import defaultdict
import copy
import logging
import random
from typing import Union

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines.tractogram import (PerArrayDict, PerArraySequenceDict)
import numpy as np
from scipy.stats import truncnorm

from dwi_ml.data.processing.streamlines.utils import split_array_at_lengths


def add_noise_to_streamlines(sft: StatefulTractogram, gaussian_size: float,
                             gaussian_variability: float,
                             noise_rng: np.random.RandomState,
                             step_size: float = None):
    """Add gaussian noise (truncated to +/- 2*noise_sigma) to streamlines
    coordinates.

    Keep in mind that you need to choose noise small enough so that directions
    won't flip. If step_size is provided, we make this check for you, ensuring
    that that noise is < step_size/2 at each step. In the worst case, the
    starting point of a segment may advance of step_size/2 while the ending
    point rewinds of step_size/2, but not further, so the direction of the
    segment won't flip. If you don't provide a step_size, make sure that your
    noise_sigma<step_size/4. Then, the maximum noise would be <step_size/2.

    Parameters
    ----------
    sft : StatefulTractogram
        Streamlines.
    gaussian_size : float
        Standard deviation of the gaussian noise to add to the streamlines. If
        you don't know what to choose, 0.1 or 0.1*step_size are both good
        approximations.
        CAREFUL. We do not deal with space. Make sure your noise is in the
        same space as your sft.
        CAREFUL.  Default: 0.
    gaussian_variability: float
        If this is given, a variation is applied to the gaussian_size to have
        more noisy streamlines and less noisy streamlines. This means that the
        real gaussian_size will be a random number between
        [size - variability, size + variability]. Default: 0.
    noise_rng : np.random.RandomState object
        Random number generator.
    step_size: float
        We will ensure the noise is not more than step_size/2 at each step. If
        None (because your are not resampling, for instance), be extra careful
        in the gaussian_size and gaussian_variability you provide.

    Returns
    -------
    noisy_sft : StatefulTractogram
        Noisy streamlines. Note. Adding noise may create invalid streamlines
        (i.e. out of the box in voxel space). If you want to save noisy_sft,
        please perform noisy_sft.remove_invalid_streamlines() first.
    """

    # Modify gaussian_size based on gaussian_variability
    # adding a random number between
    # [-gaussian_variability gaussian_variability]
    if gaussian_variability > gaussian_size:
        logging.warning('Gaussian variability ({}) should be smaller than '
                        'Gaussian size ({}) to avoid negative noise.'
                        .format(gaussian_variability, gaussian_size))
    gaussian_size = random.uniform(gaussian_size - gaussian_variability,
                                   gaussian_size + gaussian_variability)

    # Perform noise addition (flattening before to go faster)
    # max: min(2*gaussian_size, step_size/2)
    #    = min(2, step_size/(2*gaussian_size)) * gaussian_size
    flattened_coords = np.concatenate(sft.streamlines, axis=0)
    if step_size:
        max_noise_unscaled = min(2., step_size / (2 * gaussian_size))
    else:
        max_noise_unscaled = 2

    # Adding noise: between
    # (-max_noise_unscaled, max_noise_unscaled) * gaussian_size
    flattened_coords += truncnorm.rvs(-max_noise_unscaled, max_noise_unscaled,
                                      size=flattened_coords.shape,
                                      scale=gaussian_size,
                                      random_state=noise_rng)
    noisy_streamlines = split_array_at_lengths(
        flattened_coords, [len(s) for s in sft.streamlines])

    # Create output tractogram
    noisy_sft = StatefulTractogram.from_sft(
        noisy_streamlines, sft, data_per_point=sft.data_per_point,
        data_per_streamline=sft.data_per_streamline)

    return noisy_sft


def split_streamlines(sft: StatefulTractogram, rng: np.random.RandomState,
                      split_ids: np.ndarray = None, min_nb_points: int = 12):
    """Splits (or cuts) streamlines into 2 random segments. Returns both
    segments as independent streamlines, so the number of output streamlines is
    higher than the input.

    Note. Data_per_point is cut too. Data_per_streamline is copied for each
    half.

    Params
    ------
    sft: StatefulTractogram
        Dipy object containing your streamlines
    rng: np.random.RandomState
        Random number generator.
    split_ids: np.ndarray, optional
        List of streamlines to split. If not provided, all streamlines are
        split.
    min_nb_points: int
        Only cut streamlines with at least min_nb_points. This means that the
        minimal number of points in the final streamlines will be
        floor(min_nb_points/2). Default: 6.

    Returns
    -------
    new_sft: StatefulTractogram
        Dipy object with split streamlines.
    """
    if split_ids is None:
        split_ids = range(len(sft.streamlines))

    min_final_nb_points = np.floor(min_nb_points / 2)

    all_streamlines = []
    all_dpp = defaultdict(lambda: [])
    all_dps = defaultdict(lambda: [])

    # Splitting len(split_ids) streamlines out of len(sft.streamlines)
    # (If selected IDs are not too short)
    for i in range(len(sft.streamlines)):
        old_streamline = sft.streamlines[i]
        old_dpp = sft.data_per_point[i]
        old_dps = sft.data_per_streamline[i]

        # Cut if at least min_nb_points
        if i in split_ids:
            if len(old_streamline) > min_nb_points:
                cut_idx = rng.randint(
                    min_final_nb_points,
                    len(old_streamline) - min_final_nb_points)
                segments_s = [old_streamline[:cut_idx],
                              old_streamline[cut_idx:]]
                segments_dpp = [old_dpp[:cut_idx], old_dpp[cut_idx:]]

                all_streamlines.extend(segments_s)
                all_dpp = _extend_dict(all_dpp, segments_dpp[0])
                all_dpp = _extend_dict(all_dpp, segments_dpp[1])
                all_dps = _extend_dict(all_dps, old_dps)
                all_dps = _extend_dict(all_dps, old_dps)
        else:
            all_streamlines.extend([old_streamline])
            all_dpp = _extend_dict(all_dpp, old_dpp)
            all_dps = _extend_dict(all_dps, old_dps)

    new_sft = StatefulTractogram.from_sft(all_streamlines, sft,
                                          data_per_point=all_dpp,
                                          data_per_streamline=all_dps)

    return new_sft


def _extend_dict(main_dict: Union[PerArraySequenceDict, PerArrayDict],
                 added_dict: Union[PerArraySequenceDict, PerArrayDict]):
    """
    We can't do anything like main_dpp.extend(added_dpp).
    Doing as in nibabel.streamlines.tests.test_tractogram.
    """
    for k, v in added_dict.items():
        main_dict[k].append(v)
    return main_dict


def reverse_streamlines(sft: StatefulTractogram,
                        reverse_ids: np.ndarray = None):
    """Reverse streamlines, i.e. inverse the beginning and end

    Parameters
    ----------
    sft: StatefulTractogram
        Dipy object containing your streamlines
    reverse_ids: np.ndarray, optional
        List of streamlines to reverse. If not provided, all streamlines are
        reversed.

    Returns
    -------
    new_sft: StatefulTractogram
        Dipy object with reversed streamlines and data_per_point.
    """
    if reverse_ids is None:
        reverse_ids = range(len(sft.streamlines))

    new_streamlines = [s[::-1] if i in reverse_ids else s for i, s in
                       enumerate(sft.streamlines)]
    new_data_per_point = copy.deepcopy(sft.data_per_point)
    for key in sft.data_per_point:
        new_data_per_point[key] = [d[::-1] if i in reverse_ids else d for i, d
                                   in enumerate(new_data_per_point[key])]

    new_sft = StatefulTractogram.from_sft(
        new_streamlines, sft, data_per_point=new_data_per_point,
        data_per_streamline=sft.data_per_streamline)

    return new_sft
