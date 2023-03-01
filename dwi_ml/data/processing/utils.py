# -*- coding: utf-8 -*-

import logging
from typing import List

import torch


def add_noise_to_tensor(batch_data: List[torch.Tensor], gaussian_size: float,
                        gaussian_variability: float, device=None):
    """
    Add gaussian noise to data: normal distribution centered at 0,
    with sigma=gaussian_size. Noise is truncated at +/- 2*gaussian_size.

    Gaussian variability changes the gaussian noise for each tensor.

    Parameters
    ----------
    batch_data : List[Tensor]
        Batch of data tensors to which to add noise.
    gaussian_size : float
        Standard deviation of the gaussian noise to add to the tensors.
    gaussian_variability: float
        If this is given, a variation is applied to the gaussian_size to have
        more noisy streamlines and less noisy streamlines. This means that the
        real gaussian_size will be a random number between
        [size - variability, size + variability]. Default: 0.
    device: torch device

    Returns
    -------
    noisy_batch : List[Tensor]
        Noisy data.
        Note. Adding noise to streamlines may create invalid streamlines
        (i.e. out of the box in voxel space). If you want to save a noisy sft,
        please perform noisy_sft.remove_invalid_streamlines() first.
    """
    batch_size = len(batch_data)
    each_tensor_size = [len(d) for d in batch_data]

    if gaussian_variability == 0:
        # Flattening to go faster
        flattened_batch = torch.cat(batch_data, dim=0)
        noise = torch.normal(mean=0., std=gaussian_size,
                             size=flattened_batch.shape, device=device)
        max_noise = 2 * gaussian_size
        flattened_batch += torch.clip(noise, -max_noise, max_noise)
        noisy_data = torch.split(flattened_batch, each_tensor_size, dim=0)
    else:
        # Modify gaussian_size based on variability, for each tensor: adding a
        # random number between [-gaussian_variability gaussian_variability].
        # Ex: if this is a batch of streamlines, we don't want the whole batch
        # to be more noisy. Some streamlines more, some streamlines less.
        if gaussian_variability > gaussian_size:
            logging.warning('Gaussian variability ({}) should be smaller than '
                            'Gaussian size ({}) to avoid negative noise.'
                            .format(gaussian_variability, gaussian_size))
        gaussian_vars = (2 * torch.rand(batch_size) - 1) * gaussian_variability
        gaussian_sizes = gaussian_size + gaussian_vars
        noisy_data = [None] * batch_size
        for i in range(batch_size):
            noise = torch.normal(
                mean=0., std=gaussian_sizes[i],
                size=batch_data[i].shape, device=device)
            max_noise = 2 * gaussian_sizes[i]
            noisy_data[i] = batch_data[i] + \
                torch.clip(noise, -max_noise, max_noise)

    return noisy_data
