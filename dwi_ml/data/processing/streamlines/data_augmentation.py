# -*- coding: utf-8 -*-

from collections import defaultdict
import logging
from typing import Union

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines.tractogram import (PerArrayDict, PerArraySequenceDict)
import numpy as np
from scipy.stats import truncnorm

from dwi_ml.data.processing.streamlines.utils import split_array_at_lengths


# Checked!
def add_noise_to_streamlines(sft: StatefulTractogram,
                             noise_sigma: float,
                             noise_rng: np.random.RandomState):
    """Add gaussian noise (truncated to +/- 2*noise_sigma) to streamlines
    coordinates.

    Parameters
    ----------
    sft : StatefulTractogram
        Streamlines.
    noise_sigma : float
        Standard deviation of the gaussian noise to add to the streamlines.
        CAREFUL. We do not deal with space. Make sure your noise is in the
        same space as your sft.
        CAREFUL. Keep in mind that you need to choose noise_sigma<step_size/4.
        Then, the maximum noise would be <step_size/2. So in the worst case,
        the starting point of a segment may advance of step_size/2 while the
        ending point rewinds of step_size/2, but not further, so the direction
        of the segment won't flip.
    noise_rng : np.random.RandomState object
        Random number generator.

    Returns
    -------
    noisy_sft : StatefulTractogram
        Noisy streamlines. Note. Adding noise may create invalid streamlines
        (i.e. out of the box in voxel space). If you want to save noisy_sft,
        please perform noisy_sft.remove_invalid_streamlines() first.
    """
    logging.info("Please note your sft space is in {}. We suppose that the "
                 "noise, {}, fits.".format(sft.space, noise_sigma))
    
    # Perform noise addition (flattening before to go faster)
    flattened_coords = np.concatenate(sft.streamlines, axis=0)
    flattened_coords += truncnorm.rvs(-2, 2, size=flattened_coords.shape,
                                      scale=noise_sigma,
                                      random_state=noise_rng)
    noisy_streamlines = split_array_at_lengths(
        flattened_coords, [len(s) for s in sft.streamlines])

    # Create output tractogram
    noisy_sft = StatefulTractogram.from_sft(
        noisy_streamlines, sft, data_per_point=sft.data_per_point,
        data_per_streamline=sft.data_per_streamline)

    return noisy_sft


# Checked!
def split_streamlines(sft: StatefulTractogram, rng: np.random.RandomState,
                      split_ids: np.ndarray = None, min_nb_points: int = 6):
    """Cut streamlines into 2 random segments. Returns both segments as
    independent streamlines, so the number of output streamlines is higher than
    the input.

    Params
    ------
    sft: StatefulTractogram
        Dipy object containing your streamlines
    rng: np.random.RandomState
        Random number generator.
    flip_ids: np.ndarray, optional
        List of streamlines to split. If not provided, all streamlines are
        split.
    min_nb_points: int
        We only cut streamlines with at least min_nb_points. This means that the
        minimal number of points in the final streamlines is
        floor(min_nb_points/2). Default: 6.

    Returns
    -------
    new_sft: StatefulTractogram
        Dipy object with cut streamlines. Data_per_point is cut too.
        Data_per_streamline is copied for each half.
    """
    if split_ids is None:
        split_ids = range(len(sft.streamlines))

    min_index = np.floor(min_nb_points/2)

    all_streamlines = []
    all_dpp = defaultdict(lambda: [])
    all_dps = defaultdict(lambda: [])
    for i in range(len(sft.streamlines)):
        old_streamline = sft.streamlines[i]
        old_dpp = sft.data_per_point[i]
        old_dps = sft.data_per_streamline[i]

        # Leave at least min_nb_points
        if i in split_ids and len(old_streamline) > min_nb_points:
            cut_idx = rng.randint(min_index, len(old_streamline) - min_index)
            segments_s = [old_streamline[:cut_idx], old_streamline[cut_idx:]]
            segments_dpp = [old_dpp[:cut_idx], old_dpp[cut_idx:]]

            all_streamlines.extend(segments_s)
            all_dpp = _extend_dict(all_dpp, segments_dpp[0])
            all_dpp = _extend_dict(all_dpp, segments_dpp[1])
            all_dps = _extend_dict(all_dps, old_dps)
            all_dps = _extend_dict(all_dps, old_dps)
        else:
            all_streamlines.extend(old_streamline)
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


def flip_streamlines():
    raise NotImplementedError
