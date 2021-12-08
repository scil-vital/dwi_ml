# -*- coding: utf-8 -*-
import torch


def compute_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                            unexisting_val=None, device=torch.device('cpu')):
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
    # Compute previous directions
    if nb_previous_dirs == 0:
        return []

    if unexisting_val is None:
        # toDo See what to do when values do not exist. See discussion here.
        #  https://stats.stackexchange.com/questions/169887/classification-with-partially-unknown-data
        unexisting_val = torch.zeros((1, 3), dtype=torch.float32,
                                     device=device)

    n_previous_dirs = [
        torch.cat([
            torch.cat(
                (unexisting_val.repeat(min(len(dirs) + 1, i), 1),
                 dirs[:len(dirs) - i + 1]))
            for i in range(1, nb_previous_dirs + 1)], dim=1)
        for dirs in streamlines_dirs
    ]

    return n_previous_dirs


def compute_and_normalize_directions(streamlines, device=torch.device('cpu'),
                                     normalize_directions: bool = True):
    """
    Params
    ------
    batch_streamlines: list[np.array]
            The streamlines (after data augmentation)
    device: torch device
    normalize_directions: bool
    """
    # Getting directions
    batch_directions = [torch.as_tensor(s[1:] - s[:-1],
                                        dtype=torch.float32,
                                        device=device)
                        for s in streamlines]

    # Normalization:
    if normalize_directions:
        batch_directions = [s / torch.sqrt(torch.sum(s ** 2, dim=-1,
                                                     keepdim=True))
                            for s in batch_directions]

    return batch_directions
