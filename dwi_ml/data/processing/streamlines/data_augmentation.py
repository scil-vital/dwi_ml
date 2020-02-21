from typing import List, Union

import nibabel as nib
import numpy as np

from dipy.io.stateful_tractogram import StatefulTractogram
from scipy.stats import truncnorm
from dwi_ml.data.processing.streamlines.utils import split_array_at_lengths


# Checked!
def add_noise_to_streamlines(sft: StatefulTractogram,
                             noise_sigma: float,
                             noise_rng: np.random.RandomState):
    """Add gaussian noise (truncated to +/- 2*std) to streamlines coordinates

    Parameters
    ----------
    sft : StatefulTractogram
        Streamlines.
    noise_sigma : float
        Standard deviation of the gaussian noise to add to the streamlines (in
        mm)
    noise_rng : np.random.RandomState object
        Random number generator.

    Returns
    -------
    noisy_sft : StatefulTractogram
        Noisy streamlines.
    """
    # Go to world space
    orig_space = sft.space
    sft.to_rasmm()

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
    noisy_sft.to_space(orig_space)

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