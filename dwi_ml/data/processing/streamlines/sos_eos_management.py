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
import torch
from torch.nn.functional import one_hot

from dwi_ml.data.spheres import TorchSphere


def convert_dirs_to_class(batch_dirs, sphere: TorchSphere, smooth_label=False,
                          add_sos=False, add_eos=False, device=None):
    """
    Uses the points on the sphere as classes + an additional class for the
    SOS.

    batch_streamlines: should be a list of 2D tensors.
    sphere: torch sphere.
    """
    # Find class index
    # n classes ranging from 0 to n-1.
    # We need to get a tensor of shape (0, nb_points) for each streamline.
    nb_class = sphere.vertices.shape[0]
    nb_other_classes = 0
    if add_sos:
        sos_class = nb_class + 1
        nb_other_classes += 1
    if add_eos:
        eos_class = nb_class + nb_other_classes + 1
        nb_other_classes += 1

    if smooth_label:
        # See https://github.com/itaybenou/DeepTract/, in utils.train_utils.py
        batch_idx = []
        for s in batch_dirs:
            lens = torch.linalg.norm(s, dim=-1)
            dots = torch.matmul(s, sphere.vertices.T)  # Cosine similarity
            angles = torch.div(dots, lens[:, None])

            # Labels smooth is of shape nb_points x nb_class.
            labels_smooth = torch.exp(-1 * angles / 0.1)
            labels_smooth /= torch.sum(labels_smooth)

            # Equivalent of the one-hot:
            # Add 2 classes with value 0 (EOS, SOS)
            h = torch.zeros(len(s), nb_class + nb_other_classes)
            h[:, 0:nb_class] = labels_smooth

            batch_idx.append(h)
    else:
        # Convert to one-hot. Already add SOS, EOS.
        batch_idx = [sphere.find_closest(s) for s in batch_dirs]
        batch_idx = [one_hot(s.to(dtype=torch.long),
                             num_classes=nb_class + nb_other_classes
                             ).to(dtype=torch.float)
                     for s in batch_idx]

    if not (add_sos or add_eos):
        return batch_idx

    if add_sos:
        sos_token = torch.zeros(nb_class + nb_other_classes, device=device)
        sos_token[sos_class] = 1
        if not add_eos:
            batch_idx = [torch.vstack((sos_token, s)) for s in batch_idx]
    if add_eos:
        eos_token = torch.zeros(nb_class + nb_other_classes, device=device)
        eos_token[eos_class] = 1
        if not add_sos:
            batch_idx = [torch.vstack((s, eos_token)) for s in batch_idx]
    if add_sos and add_eos:
        batch_idx = [torch.vstack((sos_token, s, eos_token)) for s in batch_idx]

    return batch_idx


def add_label_as_last_dim(batch_dirs,
                          add_sos=False, add_eos=False, device=None):
    """
    batch_dirs: list of Tensors.
    """
    if not (add_sos or add_eos):
        return batch_dirs

    return [_add_label_as_last_dim_2d(s, add_sos, add_eos, device)
            for s in batch_dirs]


def _add_label_as_last_dim_2d(dirs, add_sos=False, add_eos=False, device=None):
    nb_points, nb_features = dirs.shape

    nb_new_dim = 2 if (add_sos and add_sos) else 1
    big_data = torch.zeros(nb_points + nb_new_dim, nb_features + nb_new_dim,
                           device=device)
    if add_sos:
        sos_dim = -2 if add_eos else -1
        big_data[0, sos_dim] = 1  # SOS label.

        big_data[1:1+nb_points, 0:nb_features] = dirs
    else:
        big_data[0:nb_points, 0:nb_features] = dirs

    if add_eos:
        big_data[-1, -1] = 1  # EOS label.

    return big_data


def add_zeros_sos_eos(batch_dirs,
                      add_sos=False, add_eos=False, device=None):
    """
    This should be done on embedding data.
    Batch streamlines is expected to be a 3D tensor.
    Careful. Adds the same token (0) in both positions SOS and EOS.
    """
    if not (add_sos or add_eos):
        return batch_dirs

    batch_size, _, nb_features = batch_dirs.shape

    zeros = torch.zeros(batch_size, 1, nb_features, device=device)
    if add_sos:
        if not add_eos:
            return torch.cat((zeros, batch_dirs), dim=1)
        else:
            return torch.cat((zeros, batch_dirs, zeros), dim=1)
    else:
        return torch.cat((batch_dirs, zeros), dim=1)
