# -*- coding: utf-8 -*-

"""
sos_as_label:
    Adds an initial [0,0,0,1] direction at the first point.
    Other points become [x, y, z, 0]. Same for EOS.
sos_as_zeros: bool
    Adds a [0,0,...,0] value at the first position.
    Intended to be used after the embedding layer: your model can learn the
    embedded "token" for SOS is zeros.
sos_as_class: bool
    Convert all input directions to classes on the sphere. An additional class
    is added as SOS.
"""
from typing import List

import torch
from torch.nn.functional import one_hot, pad

from dwi_ml.data.spheres import TorchSphere


def convert_dirs_to_class(batch_dirs: List[torch.Tensor],
                          sphere: TorchSphere, smooth_labels=False,
                          add_sos=False, add_eos=False, to_one_hot=False):
    """
    Uses the points on the sphere as classes + an additional class for the
    SOS.

    batch_dirs: should be a list of 2D tensors.
    sphere: torch sphere.
    smooth_label: If true, uses smoothing like in Deeptract (Benou 2019)
    add_sos: If true, adds a class for SOS, and adds a token at the beggining
        of the streamlines.
    add_eos: Idem at the end of streamlines.
    to_one_hot: If true, converts results to one-hot vectors. Needs to be true
        with smooth_label, by default.

    Returns:
        if one_hot: Tensor of shape [nb_points, nb_class]
        else: Tensor of shape [nb_points,]
    """
    # Find class index
    # n classes ranging from 0 to n-1 for the "real" directions.
    # We need to get a tensor of shape (0, nb_points) for each streamline.
    nb_class = sphere.vertices.shape[0]
    nb_other_classes = 0
    sos_class = None
    eos_class = None
    if add_sos:
        sos_class = nb_class + 1
        nb_other_classes += 1
    if add_eos:
        eos_class = nb_class + nb_other_classes + 1
        nb_other_classes += 1

    if smooth_labels:
        # See https://github.com/itaybenou/DeepTract/, in utils.train_utils.py

        if not to_one_hot:
            raise ValueError("With smooth label, we must convert to one-hot "
                             "vectors.")
        batch_idx = []
        for s in batch_dirs:
            lens = torch.linalg.norm(s, dim=-1)
            dots = torch.matmul(s, sphere.vertices.T)  # Cosine similarity
            angles = torch.div(dots, lens[:, None])

            # Labels smooth is of shape nb_points x nb_class.
            labels_smooth = torch.exp(-1 * angles / 0.1)
            labels_smooth /= torch.sum(labels_smooth)

            if add_sos or add_eos:
                # Adding n points and n classes, n = 1 or 2.
                labels_smooth = pad(labels_smooth,
                                    (0, nb_other_classes, add_sos, add_eos))
                if add_sos:
                    labels_smooth[0, sos_class - 1] = 1.0
                if add_eos:
                    labels_smooth[-1, eos_class - 1] = 1.0

            batch_idx.append(labels_smooth)
    else:
        batch_idx = [sphere.find_closest(s) for s in batch_dirs]

        device = batch_dirs[0].device
        if add_sos:
            sos_class = torch.as_tensor(sos_class - 1, device=device)
        if add_eos:
            eos_class = torch.as_tensor(eos_class - 1, device=device)

        if add_sos and add_eos:
            batch_idx = [torch.hstack((sos_class, s, eos_class)) for s in batch_idx]
        elif add_sos:
            batch_idx = [torch.hstack((sos_class, s)) for s in batch_idx]
        elif add_eos:
            batch_idx = [torch.hstack((s, eos_class)) for s in batch_idx]

        if to_one_hot:
            batch_idx = [one_hot(s.to(dtype=torch.long),
                                 num_classes=nb_class + nb_other_classes
                                 ).to(dtype=torch.float)
                         for s in batch_idx]

    return batch_idx


def add_label_as_last_dim(batch_dirs: List[torch.Tensor],
                          add_sos=False, add_eos=False):
    """
    batch_dirs: list of Tensors.
    """
    if not (add_sos or add_eos):
        return batch_dirs

    return [_add_label_as_last_dim_2d(s, add_sos, add_eos)
            for s in batch_dirs]


def _add_label_as_last_dim_2d(dirs: torch.Tensor, add_sos, add_eos):
    nb_new_dim = 2 if (add_sos and add_eos) else 1
    # Pad = (last_dim_left, last_dim_right, first_dim_left, first_dim_right)
    #     = (   0 ,         new features,   first point, last point)
    dirs = pad(dirs, (0, nb_new_dim, add_sos, add_eos))

    if add_sos:
        sos_dim = -2 if add_eos else -1
        dirs[0, sos_dim] = 1  # SOS label.

    if add_eos:
        dirs[-1, -1] = 1  # EOS label.

    return dirs


def add_zeros_sos_eos(batch_dirs: List[torch.Tensor],
                      add_sos=False, add_eos=False):
    """
    This should be done on embedding data.
    Careful. Adds the same token (0) in both positions SOS and EOS.
    """
    if not (add_sos or add_eos):
        return batch_dirs

    if add_sos:
        if not add_eos:
            return [pad(s, (0, 0, 1, 0)) for s in batch_dirs]
        else:
            return [pad(s, (0, 0, 1, 1)) for s in batch_dirs]
    else:
        return [pad(s, (0, 0, 0, 1)) for s in batch_dirs]
