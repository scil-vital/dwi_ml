# -*- coding: utf-8 -*-

import numpy as np

from dipy.tracking.streamline import set_number_of_points

import torch


def _autoencode_streamlines(model, sft, batch_size, normalize, device):
    """
    Autoencode streamlines using the given model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for autoencoding.
    batch_size : int
        The batch size to use for encoding.
    bundle : list of np.ndarray
        The streamlines to encode.
    normalize : bool
        Whether to normalize the streamlines before encoding.
    sft : StatefulTractogram
        The stateful tractogram containing the streamlines.
    device : torch.device
        The device to use for encoding.

    Returns
    -------
    generator
        A generator that yields the encoded streamlines.
    """

    batches = range(0, len(sft.streamlines), batch_size)
    for i, b in enumerate(batches):
        with torch.no_grad():
            s = np.asarray(
                set_number_of_points(
                     sft.streamlines[i * batch_size:(i+1) * batch_size],
                     256))
            if normalize:
                s /= sft.dimensions

            streamlines = torch.as_tensor(
                s, dtype=torch.float32, device=device)
            tmp_outputs = model(streamlines).cpu().numpy()

            scaling = sft.dimensions if normalize else 1.0
            streamlines_output = tmp_outputs.transpose((0, 2, 1)) * scaling
            for strml in streamlines_output:
                yield strml, strml[0]
