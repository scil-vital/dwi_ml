# -*- coding: utf-8 -*-
import logging
from typing import List, Union

from dipy.io.stateful_tractogram import StatefulTractogram
import nibabel as nib
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


def cut_random_streamlines(
        streamlines: Union[List, nib.streamlines.ArraySequence],
        split_percentage: float, rng: np.random.RandomState):
    """Cut a percentage of streamlines into 2 random segments.
    Returns both segments as independent streamlines, so the number of
    output streamlines is higher than the input.

    Parameters
    ----------
    streamlines : nib.stramlines.ArraySequence or list
        Streamlines to cut.
    split_percentage : float
        Percentage of streamlines to cut.
    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    output_streamlines : nib.stramlines.ArraySequence or list
        Result of cutting.
    """
    all_ids = np.arange(len(streamlines))
    n_to_split = int(np.floor(len(streamlines) * split_percentage))
    split_ids = rng.choice(all_ids, size=n_to_split, replace=False)

    output_streamlines = []
    for i, s in enumerate(streamlines):
        # Leave at least 6 points
        if i in split_ids and len(s) > 6:
            cut_idx = rng.randint(3, len(s) - 3)
            segments = [s[:cut_idx], s[cut_idx:]]
        else:
            segments = [s]
        output_streamlines.extend(segments)

    return output_streamlines


def flip_streamlines():
    raise NotImplementedError
