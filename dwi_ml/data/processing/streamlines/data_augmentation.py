# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
import copy
from typing import Union

from dipy.io.stateful_tractogram import StatefulTractogram
from nibabel.streamlines.tractogram import (PerArrayDict, PerArraySequenceDict)
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft


def resample_or_compress(sft, step_size_mm: float = None,
                         compress: float = None):
    if step_size_mm is not None:
        # Note. No matter the chosen space, resampling is done in mm.
        logging.debug("            Resampling: {}".format(step_size_mm))
        sft = resample_streamlines_step_size(sft, step_size=step_size_mm)
    if compress is not None:
        logging.debug("            Compressing: {}".format(compress))
        sft = compress_sft(sft, compress)
    return sft


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
    Doing as in nibabel.streamlines.utils.test_tractogram.
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
