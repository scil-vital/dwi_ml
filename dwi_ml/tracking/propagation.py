# -*- coding: utf-8 -*-
import logging
from typing import Callable, List

import numpy as np
import torch
from torch import Tensor

from dwi_ml.tracking.tracking_mask import TrackingMask

logger = logging.getLogger('tracker_logger')


def propagate_multiple_lines(
        lines: List[Tensor], update_memory_after_removing_lines: Callable,
        get_next_dirs: Callable, theta: float, step_size: float,
        verify_opposite_direction: bool = False,
        mask: TrackingMask = None, max_nbr_pts: int = None,
        append_last_point: bool = True, normalize_directions: bool = True):
    """
    Propagates initialized streamlines.

    Parameters
    ----------
    lines: List[Tensor]
        Streamlines. Either they are ALL only one point (the seed point), or
        they ALL contain more than one point and we propagate.
    update_memory_after_removing_lines: Callable
        A function with format:
        None = update_memory_after_removing_lines(
        can_continue: np.ndarray, new_stopping_lines_raw_idx: List).
        In case you need to update some internal states.
    get_next_dirs: Callable
        A function with format:
        next_dirs = get_next_dirs(lines: List[Tensor], n_last_pos: List[Tensor])
    theta: float
    step_size: float
    verify_opposite_direction: bool
    mask: TrackingMask
    max_nbr_pts: int
    append_last_point: bool
    normalize_directions: bool
    """
    nb_streamlines = len(lines)

    # Monitoring
    invalid_direction_counts = np.zeros(nb_streamlines)
    continuing_lines_rawidx = np.arange(nb_streamlines)

    # Will get the final lines when they are done.
    final_lines = [None] * nb_streamlines  # type: List[Tensor]

    # Find initial direction
    all_lines_completed = False
    if len(lines[0]) == 1:
        # Starting from zero. We suppose all streamlines are starting from
        # zero.
        previous_dir = None
    else:
        previous_dir = [line[-1, :] - line[-2, :] for line in lines]
        previous_dir = torch.vstack(previous_dir)
        if normalize_directions:
            previous_dir /= torch.linalg.norm(previous_dir, dim=-1)[:, None]

    # Track
    while not all_lines_completed:
        n_new_pos, previous_dir, invalid_dirs = \
            _take_one_step_or_go_straight(
                lines, previous_dir, get_next_dirs, theta, step_size,
                normalize_directions, verify_opposite_direction)

        # If invalid direction (ex: angle or EOS), stop now.
        if sum(invalid_dirs) > 0:
            logger.debug("{} streamlines with invalid directions "
                         "(ex, EOS, angle).".format(sum(invalid_dirs)))

        # For other streamlines: verifying but appending only if option is
        # chosen.
        break_with_appending = _verify_stopping_criteria(
            n_new_pos, lines, mask, max_nbr_pts)

        if append_last_point:
            # Appending last point only to streamlines with valid dir (i.e.
            # no wrong angle or NaN direction, ex due to EOS).
            lines = [torch.vstack((s, n_new_pos[i, :])) if ~invalid_dirs[i]
                     else s for i, s in enumerate(lines)]
            breaking_now = np.logical_or(break_with_appending, invalid_dirs)
            can_continue = ~breaking_now
        else:
            # Appending last point only to continuing streamlines, i.e. valid
            # dirs + no stopping criteria.
            breaking_now = np.logical_or(break_with_appending, invalid_dirs)
            can_continue = ~breaking_now
            lines = [torch.vstack((s, n_new_pos[i, :])) if can_continue[i]
                     else s for i, s in enumerate(lines)]

        # Saving finished streamlines
        idx_stop, = np.where(breaking_now)
        for i in idx_stop:
            final_lines[continuing_lines_rawidx[i]] = lines[i]

        # Update model if needed.
        if np.any(breaking_now):
            new_stopping_lines_raw_idx = continuing_lines_rawidx[breaking_now]
            update_memory_after_removing_lines(
                can_continue, new_stopping_lines_raw_idx)

        # Keeping only remaining lines.
        if np.any(can_continue):
            lines = [s for i, s in enumerate(lines) if can_continue[i]]
            previous_dir = previous_dir[can_continue, :]
            continuing_lines_rawidx = continuing_lines_rawidx[can_continue]
            invalid_direction_counts = invalid_direction_counts[can_continue]
        else:
            all_lines_completed = True

    assert not np.any([line is None for line in final_lines])

    return final_lines


def _take_one_step_or_go_straight(
        lines: List[Tensor], previous_dirs: Tensor,
        get_next_dirs: Callable, theta: float, step_size: float,
        normalize_directions: bool = True,
        verify_opposite_direction: bool = False):
    """
    Finds the next direction. If no valid direction is found (invalid = if
    the model returns NaN, ex if EOS is used, or if the angle is too
    sharp). Then, the previous direction is copied but the list of invalid
    directions is returned.

    Return
    ------
    n_new_pos: Tensor(n, 3)
        The new positions.
    next_dirs: Tensor(n, 3)
        The new segment direction. The previous direction is copied if no
        valid direction is found. Normalized if normalize_directions.
    invalid_dirs: ndarray(n, )
        True if new_dir is invalid.
    """
    last_pos = [line[-1, :] for line in lines]
    next_dirs = get_next_dirs(lines, last_pos)

    if isinstance(next_dirs, list):
        next_dirs = torch.vstack(next_dirs)

    if normalize_directions:
        next_dirs /= torch.linalg.norm(next_dirs, dim=-1)[:, None]

    if previous_dirs is not None:
        # Verify angle
        next_dirs = _verify_angle(
            next_dirs, previous_dirs, theta,
            already_normalized=normalize_directions,
            verify_opposite_direction=verify_opposite_direction)

        # Go straight if we got no next direction.
        invalid_dirs = torch.isnan(next_dirs[:, 0]).cpu().numpy()
        next_dirs[invalid_dirs, :] = previous_dirs[invalid_dirs, :]
    else:
        invalid_dirs = np.zeros(len(next_dirs), dtype=bool)

    # Get new positions
    last_pos = torch.vstack(last_pos)
    n_new_pos = last_pos + step_size * next_dirs

    return n_new_pos, next_dirs, invalid_dirs


def _verify_stopping_criteria(n_last_pos, lines, mask=None, max_nbr_pts=None):
    """
    mask can be None, or if you want to check bounds, you can set an empty mask
    (with mask.data = None).
    """

    # Checking total length. During forward: all the same length. Not
    # during backward.
    if max_nbr_pts is not None:
        stopping = np.asarray([len(s) for s in lines]) == max_nbr_pts
        if sum(stopping) > 0:
            logger.debug("{} streamlines stopping after reaching max nb "
                         "points ({})".format(sum(stopping), max_nbr_pts))
    else:
        stopping = np.zeros(len(lines), dtype=bool)

    # Checking if out of bound using seeding mask
    if mask is not None:
        out_of_mask = ~mask.is_vox_corner_in_bound(n_last_pos).cpu().numpy()
        if sum(out_of_mask) > 0:
            logger.debug("{} streamlines stopping out of bounds."
                         .format(sum(out_of_mask)))
        stopping = np.logical_or(stopping, out_of_mask)

        if mask.data is not None and not np.all(stopping):
            # Checking if out of mask
            # Avoid interpolation for points that we already know can't
            # continue.
            still_on = ~stopping

            out_of_mask = ~mask.is_in_mask(n_last_pos[still_on]).cpu().numpy()
            if sum(out_of_mask) > 0:
                logger.debug("{} streamlines stopping out of mask."
                             .format(sum(out_of_mask)))
            stopping[still_on] = out_of_mask

    return stopping


def _verify_angle(next_dirs: Tensor, previous_dirs: Tensor, theta,
                  already_normalized=False, verify_opposite_direction=False):
    # toDo could we find a better solution for proba tracking?
    #  Resampling until angle < theta? Easy on the sphere (restrain
    #  probas on the sphere pics inside a cone theta) but what do we do
    #  for other models? Ex: For the Gaussian direction getter?
    if already_normalized:
        cos_angle = torch.sum(next_dirs * previous_dirs, dim=1)
    else:
        norm1 = torch.linalg.norm(next_dirs, dim=-1)
        norm2 = torch.linalg.norm(previous_dirs, dim=-1)
        cos_angle = torch.sum(
            torch.div(next_dirs, norm1[:, None]) *
            torch.div(previous_dirs, norm2[:, None]), dim=1)

    # Resolving numerical instabilities:
    # (Converts angle to numpy)
    # Torch does not have a min() for tensor vs scalar. Using np. Ok,
    # small step.
    one = torch.ones(1, device=next_dirs.device)
    cos_angle = torch.minimum(torch.maximum(-one, cos_angle), one)
    angles = torch.arccos(cos_angle)

    if verify_opposite_direction:
        mask_angle = angles > np.pi / 2  # 90 degrees
        angles[mask_angle] = np.mod(angles[mask_angle] + np.pi, 2*np.pi)
        next_dirs[mask_angle] = - next_dirs[mask_angle]

    mask_angle = angles > theta
    next_dirs[mask_angle] = torch.full((3,), fill_value=torch.nan,
                                       device=next_dirs.device)

    return next_dirs
