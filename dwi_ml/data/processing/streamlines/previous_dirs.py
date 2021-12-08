# -*- coding: utf-8 -*-
import torch


def compute_n_previous_dirs(streamlines_dirs, unexisting_val,
                            nb_previous_dirs):
    """
    Params
    ------
    streamline_dirs: list[torch.tensor]
        A list of length nb_streamlines. Each tensor is of size
        [(N-1) x 3]; where N is the length of the streamline.
        If one streamline contains no dirs (tensor = []), by default, we
        consider it had one point (with no existing previous_dirs).
    unexisting_val: torch.tensor:
        Tensor to use as n^th previous direction when this direction does not
        exist (ex: 2nd previous direction for the first point).
        Ex: torch.zeros((1, 3))

    Returns
    -------
    previous_dirs: list[tensor]
        A list of length nb_streamlines. Each tensor is of size
        [N, nb_previous_dir x 3]; the n previous dirs at each
        point of the streamline. Order is 1st previous dir, 2nd previous dir,
        etc., (reading the streamline backward).
    """
    """
    # Equivalent as a loop:
    
    n_previous_dirs = []
    for dirs in streamlines_dirs:
        streamline_n_previous_dirs = []

        for i in range(1, nb_previous_dirs + 1):
            # i points have non-existing ith previous dir.
            # The last i-1 directions are not the ith previous dir of any point
            # and thus not used (meaning we use up to -(i-1) = -i + 1).
            # Could use dirs[:-i+1] but bugs when i=1: accessing [:-0] does not
            # work.
            streamline_ith_prev_dirs = torch.cat(
                (unexisting_val.repeat(min(len(dirs) + 1, i), 1),
                 dirs[:len(dirs) - i + 1]))
            streamline_n_previous_dirs.append(streamline_ith_prev_dirs)
        n_previous_dirs.append(
            torch.cat(tuple(streamline_n_previous_dirs), dim=1))
    """

    n_previous_dirs = [
        torch.cat([
            torch.cat(
                (unexisting_val.repeat(min(len(dirs) + 1, i), 1),
                 dirs[:len(dirs) - i + 1]))
            for i in range(1, nb_previous_dirs + 1)], dim=1)
        for dirs in streamlines_dirs
    ]

    return n_previous_dirs
