# -*- coding: utf-8 -*-
import logging
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scilpy.tractanalysis.connectivity_segmentation import \
    extract_longest_segments_from_profile as segmenting_func
from scilpy.tractograms.uncompress import streamlines_to_voxel_coordinates

# We could try using nan instead of zeros for non-existing previous dirs...
DEFAULT_UNEXISTING_VAL = torch.zeros((1, 3), dtype=torch.float32)


def compute_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                            unexisting_val=DEFAULT_UNEXISTING_VAL,
                            point_idx=None):
    """
    Params
    ------
    streamline_dirs: list[torch.tensor]
        A list of length nb_streamlines with the streamline direction at each
        point. Each tensor is of size [(N-1) x 3]; where N is the length of the
        streamline. For each streamline: [dir1, dir2, dir3, ...]
        ** If one streamline contains no dirs (tensor = []), by default, we
        consider it had one point (with no existing previous_dirs).
    unexisting_val: torch.tensor:
        Tensor to use as n^th previous direction when this direction does not
        exist (ex: 2nd previous direction for the first point).
        Ex: torch.zeros((1, 3))
    device: torch device
    point_idx: int
        If given, gets the n previous directions for a given point of the
        streamline. If None, returns the previous directions at each point.
        Hint: can be -1.

    Returns
    -------
    previous_dirs: list[tensor]
        A list of length nb_streamlines. Each tensor is of size
        [N, nb_previous_dir x 3]; the n previous dirs at each
        point of the streamline. Order is 1st previous dir, 2nd previous dir,
        etc., (reading the streamline backward).
        For each streamline:
             [[dirx dirx ...], ---> idx0
             [dir1 dirx ...],  ---> idx1
             [dir2 dir1 ...]]  ---> idx2
             Where dirx is a "non-existing dir" (typically, [0,0,0])
             The length of each row (...) is self.nb_previous_dirs.
    """
    device = streamlines_dirs[0].device
    if nb_previous_dirs == 0:
        return None

    unexisting_val = unexisting_val.to(device, non_blocking=True)

    if point_idx:
        prev_dirs = _get_one_n_previous_dirs(
            streamlines_dirs, nb_previous_dirs, unexisting_val, point_idx)
    else:
        prev_dirs = _get_all_n_previous_dirs(streamlines_dirs,
                                             nb_previous_dirs, unexisting_val)

    return prev_dirs


def _get_all_n_previous_dirs(streamlines_dirs: List[torch.Tensor],
                             nb_previous_dirs: int,
                             unexisting_val: torch.Tensor):
    # NOT USED.
    # More intuitive than v1, but...
    # We have seen that this can be up two 12 times slower.

    previous_dirs = [None] * len(streamlines_dirs)
    for i, dirs in enumerate(streamlines_dirs):
        nb_points = len(dirs) + 1
        previous_dirs[i] = \
            torch.zeros((nb_points, nb_previous_dirs * 3),
                        device=streamlines_dirs[0].device)

        no_n_prev_dirs = dirs
        for n in range(nb_previous_dirs):
            # The n^e previous dir is just the list of dirs, shifted right
            if n > 0:
                no_n_prev_dirs = torch.cat((unexisting_val,
                                            no_n_prev_dirs[:-1, :]))
            else:
                no_n_prev_dirs = torch.cat((unexisting_val, no_n_prev_dirs))

            previous_dirs[i][:, n*3:n*3+3] = no_n_prev_dirs

    return previous_dirs


def _get_one_n_previous_dirs(streamlines_dirs, nb_previous_dirs,
                             unexisting_val, point_idx):
    # Builds horizontally.

    # Ex: if point_idx == -1:
    # i=1 -->  last dir --> dirs[point_idx-i+1]
    # But if point_idx == 5:
    # i=1 -->  dir #4 --> dirs[point_idx - i]

    n_previous_dirs = [
        torch.cat([
            dirs[point_idx - i][None, :] if (point_idx >= 0 and i <= point_idx)
            else dirs[point_idx - i + 1][None, :] if (
                    point_idx < 0 and i <= len(dirs) + 1 + point_idx)
            else unexisting_val
            for i in range(1, nb_previous_dirs + 1)], dim=1)
        for dirs in streamlines_dirs
    ]
    return n_previous_dirs


def compute_directions(streamlines):
    """
    Params
    ------
    batch_streamlines: list[np.array]
            The streamlines (after data augmentation)
    """
    if isinstance(streamlines, list):
        batch_directions = [torch.diff(s, n=1, dim=0) for s in streamlines]
    else:  # Tensor:
        batch_directions = torch.diff(streamlines, n=1, dim=0)

    return batch_directions


def normalize_directions(directions, new_norm=1.0):
    """
    Params
    ------
    directions: list[tensor]
    """
    if isinstance(directions, torch.Tensor):
        # Not using /= because if this is used in forward propagation, backward
        # propagation will fail.
        directions = directions / torch.linalg.norm(directions, dim=-1,
                                                    keepdim=True)
        directions *= new_norm
    else:
        directions = [s / torch.linalg.norm(s, dim=-1, keepdim=True) * new_norm
                      for s in directions]

    return directions


def compute_angles(line_dirs, degrees=False):
    one = torch.ones(1, device=line_dirs.device)

    line_dirs /= torch.linalg.norm(line_dirs, dim=-1, keepdim=True)
    cos_angles = torch.sum(line_dirs[:-1, :] * line_dirs[1:, :], dim=1)

    # Resolve numerical instability
    cos_angles = torch.minimum(torch.maximum(-one, cos_angles), one)
    angles = torch.arccos(cos_angles)

    if degrees:
        angles = torch.rad2deg(angles)
    return angles


def compress_streamline_values(
        streamlines: List = None, dirs: List = None, values: List = None,
        compress_eps: float = 1e-3):
    """
    Parameters
    ----------
    streamlines: List[Tensors]
        Streamlines' coordinates. If None, dirs must be given.
    dirs: List[Tensors]
        Streamlines' directions, optional. Useful to skip direction computation
        if already computed elsewhere.
    values: List[Tensors]
        If set, compresses the values rather than the streamlines themselves.
    compress_eps: float
        Angle (in degrees)
    """
    if streamlines is None and dirs is None:
        raise ValueError("You must provide either streamlines or dirs.")
    elif dirs is None:
        dirs = compute_directions(streamlines)

    if values is None:
        assert streamlines is not None
        # Compress the streamline itself with our technique.
        # toDo
        raise NotImplementedError("Code not ready")

    compress_eps = np.deg2rad(compress_eps)

    compressed_mean_loss = 0.0
    compressed_n = 0
    for loss, line_dirs in zip(values, dirs):
        if len(loss) < 2:
            compressed_mean_loss = compressed_mean_loss + torch.mean(loss)
            compressed_n += len(loss)
        else:
            # 1. Compute angles
            angles = compute_angles(line_dirs)

            # 2. Compress losses
            # By definition, the starting point is different from previous
            # and has an important meaning. Separating.
            compressed_mean_loss = compressed_mean_loss + loss[0]
            compressed_n += 1

            # Then, verifying other segments
            current_loss = 0.0
            current_n = 0
            for next_loss, next_angle in zip(loss[1:], angles):
                # toDO. Find how to skip loop
                if next_angle < compress_eps:
                    current_loss = current_loss + next_loss
                    current_n += 1
                else:
                    if current_n > 0:
                        # Finish the straight segment
                        compressed_mean_loss = \
                            compressed_mean_loss + current_loss / current_n
                        compressed_n += 1

                    # Add the point following a big curve separately
                    compressed_mean_loss = compressed_mean_loss + next_loss
                    compressed_n += 1

                    # Then restart a possible segment
                    current_loss = 0.0
                    current_n = 0

    return compressed_mean_loss / compressed_n, compressed_n


def weight_value_with_angle(values: List, streamlines: List = None,
                            dirs: List = None):
    """
    Parameters
    ----------
    values: List[Tensors]
        Value to weight with angle. Ex: losses.
    streamlines: List[Tensors]
        Streamlines' coordinates. If None, dirs must be given.
    dirs: List[Tensors]
        Streamlines' directions, optional. Useful to skip direction computation
        if already computed elsewhere.
    """
    if streamlines is None and dirs is None:
        raise ValueError("You must provide either streamlines or dirs.")
    elif dirs is None:
        dirs = compute_directions(streamlines)

    assert np.array_equal([len(v) for v in values], [len(d) for d in dirs])

    zero = torch.as_tensor(0.0, device=dirs[0].device)
    for i, line_dirs in enumerate(dirs):
        angles = compute_angles(line_dirs, degrees=True)
        # Adding a zero angle for first value.
        angles = torch.hstack([zero, angles])

        # Mult choice:
        # We don't want to multiply by 0. Multiplying by angles + 1.
        values[i] = values[i] * (angles + 1.0)

    return values


def _compute_origin_finish_blocs(streamlines, volume_size, nb_blocs):
    # Getting endpoint coordinates
    volume_size = np.asarray(volume_size)
    if isinstance(streamlines[0], list):
        start_values = [s[0] for s in streamlines]
        end_values = [s[-1] for s in streamlines]
    elif isinstance(streamlines[0], torch.Tensor):
        start_values = [s[0, :].cpu().numpy() for s in streamlines]
        end_values = [s[-1, :].cpu().numpy() for s in streamlines]
    else:  # expecting numpy arrays
        start_values = [s[0, :] for s in streamlines]
        end_values = [s[-1, :] for s in streamlines]

    # Diving into blocs (a type of downsampling)
    mult_factor = nb_blocs / volume_size
    start_values = np.clip((start_values * mult_factor).astype(int),
                           a_min=0, a_max=nb_blocs - 1)
    end_values = np.clip((end_values * mult_factor).astype(int),
                         a_min=0, a_max=nb_blocs - 1)

    # Blocs go from 0 to m1*m2*m3.
    nb_dims = len(nb_blocs)
    start_block = np.ravel_multi_index(
        [start_values[:, d] for d in range(nb_dims)], nb_blocs)

    end_block = np.ravel_multi_index(
        [end_values[:, d] for d in range(nb_dims)], nb_blocs)

    return start_block, end_block


def compute_triu_connectivity_from_labels(streamlines, data_labels,
                                          use_scilpy=False):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    streamlines: list of np arrays or list of tensors.
        Streamlines, in vox space, corner origin.
    data_labels: np.ndarray
        The loaded nifti image.
    use_scilpy: bool
        If True, uses scilpy's method:
          'Strategy is to keep the longest streamline segment
           connecting 2 regions. If the streamline crosses other gray
           matter regions before reaching its final connected region,
           the kept connection is still the longest. This is robust to
           compressed streamlines.'
        Else, uses simple computation from endpoints. Faster. Also, works with
        incomplete parcellation.

    Returns
    -------
    matrix: np.ndarray
        With use_scilpy: shape (nb_labels + 1, nb_labels + 1)
        (last label is "Not Found")
        Else, shape (nb_labels, nb_labels)
    labels: List
        The list of labels
    start_labels: List
        For each streamline, the label at starting point.
    end_labels: List
        For each streamline, the label at ending point.
    """
    real_labels = list(np.sort(np.unique(data_labels)))
    nb_labels = len(real_labels)
    logging.debug("Computing connectivity matrix for {} labels."
                  .format(nb_labels))

    if use_scilpy:
        matrix = np.zeros((nb_labels + 1, nb_labels + 1), dtype=int)
    else:
        matrix = np.zeros((nb_labels, nb_labels), dtype=int)

    start_labels = []
    end_labels = []

    if use_scilpy:
        indices, points_to_idx = streamlines_to_voxel_coordinates(
            streamlines, return_mapping=True)

        for strl_vox_indices in indices:
            segments_info = segmenting_func(strl_vox_indices, data_labels)
            if len(segments_info) > 0:
                start = real_labels.index(segments_info[0]['start_label'])
                end = real_labels.index(segments_info[0]['end_label'])
            else:
                start = nb_labels
                end = nb_labels

            start_labels.append(start)
            end_labels.append(end)

            matrix[start, end] += 1
            if start != end:
                matrix[end, start] += 1

        real_labels = real_labels + [np.NaN]

    else:
        for s in streamlines:
            # Vox space, corner origin
            # = we can get the nearest neighbor easily.
            # Coord 0 = voxel 0. Coord 0.9 = voxel 0. Coord 1 = voxel 1.
            start = real_labels.index(
                data_labels[tuple(np.floor(s[0, :]).astype(int))])
            end = real_labels.index(
                data_labels[tuple(np.floor(s[-1, :]).astype(int))])

            start_labels.append(start)
            end_labels.append(end)

            matrix[start, end] += 1
            if start != end:
                matrix[end, start] += 1

    matrix = np.triu(matrix)
    assert matrix.sum() == len(streamlines)

    return matrix, real_labels, start_labels, end_labels


def compute_triu_connectivity_from_blocs(streamlines, volume_size, nb_blocs):
    """
    Compute a connectivity matrix.

    Parameters
    ----------
    streamlines: list of np arrays or list of tensors.
        Streamlines, in vox space, corner origin.
    volume_size: list
        The 3D dimension of the reference volume.
    nb_blocs:
        The m1 x m2 x m3 = mmm number of blocs for the connectivity matrix.
        This means that the matrix will be a mmm x mmm triangular matrix.
        In 3D, with 20x20x20, this is an 8000 x 8000 matrix (triangular). It
        probably contains a lot of zeros with the background being included.
        Can be saved as sparse.
    """
    nb_blocs = np.asarray(nb_blocs)
    start_block, end_block = _compute_origin_finish_blocs(
        streamlines, volume_size, nb_blocs)

    total_size = np.prod(nb_blocs)
    matrix = np.zeros((total_size, total_size), dtype=int)
    for s_start, s_end in zip(start_block, end_block):
        matrix[s_start, s_end] += 1

        # Either, at the end, sum lower triangular + upper triangular (except
        # diagonal), or:
        if s_end != s_start:
            matrix[s_end, s_start] += 1

    matrix = np.triu(matrix)
    assert matrix.sum() == len(streamlines)

    return matrix, start_block, end_block


def prepare_figure_connectivity(matrix):
    matrix = np.copy(matrix)

    fig, axs = plt.subplots(2, 2)
    im = axs[0, 0].imshow(matrix)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 0].set_title("Raw streamline count")

    im = axs[0, 1].imshow(matrix + np.min(matrix[matrix > 0]), norm=LogNorm())
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 1].set_title("Raw streamline count (log view)")

    matrix = matrix / matrix.sum() * 100
    im = axs[1, 0].imshow(matrix)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[1, 0].set_title("Percentage")

    matrix = matrix > 0
    axs[1, 1].imshow(matrix)
    axs[1, 1].set_title("Binary")

    plt.suptitle("All versions of the connectivity matrix.")


def find_streamlines_with_chosen_connectivity(
        streamlines, start_labels, end_labels, label1, label2=None):
    """
    Returns streamlines corresponding to a (label1, label2) or (label2, label1)
    connection.

    Parameters
    ----------
    streamlines: list of np arrays or list of tensors.
        Streamlines, in vox space, corner origin.
    start_labels: list[int]
        The starting bloc for each streamline.
    end_labels: list[int]
        The ending bloc for each streamline.
    label1: int
        The bloc of interest, either as starting or finishing point.
    label2: int, optional
        The bloc of interest, either as starting or finishing point.
        If label2 is None, then all connections (label1, Y) and (X, label1)
        are found.
    """
    start_labels = np.asarray(start_labels)
    end_labels = np.asarray(end_labels)

    if label2 is None:
        labels2 = np.unique(np.concatenate((start_labels[:], end_labels[:])))
    else:
        labels2 = [label2]

    found = np.zeros(len(streamlines))
    for label2 in labels2:
        str_ind1 = np.logical_and(start_labels == label1,
                                  end_labels == label2)
        str_ind2 = np.logical_and(start_labels == label2,
                                  end_labels == label1)
        str_ind = np.logical_or(str_ind1, str_ind2)
        found = np.logical_or(found, str_ind)

    return [s for i, s in enumerate(streamlines) if found[i]]
