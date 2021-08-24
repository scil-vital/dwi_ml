# -*- coding: utf-8 -*-
from typing import List

import numpy as np


def split_array_at_lengths(array: np.ndarray, lengths: List[int]):
    """Split an array into sub-arrays, provided the length of each segment.

    Parameters
    ----------
    array : np.ndarray with shape (N, ...)
        The array to split.
    lengths : List of int
        The required length of each sub-array

    Returns
    -------
    np.ndarray with shape (len(lengths), ...)
        The split array, where the length of each sub-array corresponds to the
        provided `lengths`.
    """
    assert sum(lengths) == array.shape[0], \
        "The sum of the provided lengths does not equal the size of the " \
        "array to split!"

    # Last offset will output an empty sub-array
    offsets = np.cumsum(lengths)[:-1]

    return np.split(array, offsets)
