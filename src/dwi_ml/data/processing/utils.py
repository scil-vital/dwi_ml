# -*- coding: utf-8 -*-
from typing import List

import torch


def add_noise_to_tensor(batch_data: List[torch.Tensor], gaussian_size: float,
                        device=None):
    """
    Add gaussian noise to data: normal distribution centered at 0,
    with sigma=gaussian_size. Noise is truncated at +/- 2*gaussian_size.

    Parameters
    ----------
    batch_data : List[Tensor]
        Batch of data tensors to which to add noise.
    gaussian_size : float
        Standard deviation of the gaussian noise to add to the tensors.
    device: torch device

    Returns
    -------
    noisy_batch : List[Tensor]
        Noisy data.
        Note. Adding noise to streamlines may create invalid streamlines
        (i.e. out of the box in voxel space). If you want to save a noisy sft,
        please perform noisy_sft.remove_invalid_streamlines() first.
    """
    each_tensor_size = [len(d) for d in batch_data]

    # Flattening to go faster
    flattened_batch = torch.cat(batch_data, dim=0)
    noise = torch.normal(mean=0., std=gaussian_size,
                         size=flattened_batch.shape, device=device)
    max_noise = 2 * gaussian_size
    flattened_batch += torch.clip(noise, -max_noise, max_noise)
    # Doc of split says it returns list, but it returns tuple
    noisy_data = list(torch.split(flattened_batch, each_tensor_size, dim=0))

    return noisy_data
