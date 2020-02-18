from typing import List, Union

import nibabel as nib
import numpy as np

from scipy.stats import truncnorm
from dwi_ml.data.processing.space.convert_world_to_vox import (
    convert_world_to_vox)
from dwi_ml.data.processing.streamlines.utils import split_array_at_lengths


def add_noise_to_streamlines(
        streamlines: Union[nib.streamlines.ArraySequence, np.ndarray],
        noise_sigma: float, noise_rng: np.random.RandomState,
        convert_mm_to_vox: bool = False, affine: np.ndarray = None)\
        -> Union[nib.streamlines.ArraySequence, np.ndarray]:
    """Add gaussian noise (truncated to +/- 2*std) to streamlines coordinates
     *in-place*

    Parameters
    ----------
    streamlines : nib.streamlines.ArraySequence
        Streamlines.
    noise_sigma : float
        Standard deviation of the gaussian noise to add to the streamlines.
        NOTE: This value should be in the same space as the streamlines
        (assuming the space is isometric)
    noise_rng : np.random.RandomState object
        Random number generator.
    convert_mm_to_vox: bool
        If true, will convert noise from mm to vox_iso using affine first.
        Note that we don't support vox to mm yet, nor mm to vox_noniso.
        [False]
    affine: np.ndarray
        Needed if convert_noise_space is True. Ex : affine_vox2rasmm

    Returns
    -------
    streamlines : nib.streamlines.ArraySequence
        Noisy streamlines.
    """

    # Dealing with spaces: putting noise in the streamline's space
    if convert_mm_to_vox:
        noise_sigma = convert_world_to_vox(noise_sigma, affine)

    # Performing noise addition
    if isinstance(streamlines, nib.streamlines.ArraySequence):
        # Access to dipy protected. See if they change that.
        streamlines._data += truncnorm.rvs(-2, 2, size=streamlines.data.shape,
                                           scale=noise_sigma,
                                           random_state=noise_rng)
    elif isinstance(streamlines, np.ndarray):
        # Flattening to go faster
        flattened_coords = np.concatenate(streamlines, axis=0)

        # Add noise
        flattened_coords += truncnorm.rvs(-2, 2, size=flattened_coords.shape,
                                          scale=noise_sigma,
                                          random_state=noise_rng)

        # Unflatten
        streamlines = split_array_at_lengths(flattened_coords,
                                             [len(s) for s in streamlines])

    return streamlines


def cut_random_streamlines(
        streamlines: Union[List, nib.streamlines.ArraySequence],
        split_percentage: float, rng: np.random.RandomState):
    """Cut a percentage of streamlines in 2 random segments.
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
