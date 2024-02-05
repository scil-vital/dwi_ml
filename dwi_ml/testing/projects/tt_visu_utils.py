# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm


def reshape_unpad_rescale_attention(
        attention_per_layer, average_heads: bool, average_layers,
        group_with_max, lengths, rescale_0_1, rescale_dev):
    """
    Also sends to CPU.

    Parameters
    ----------
    attention_per_layer: List[Tensor]
        A list: nb_layers x
                   [nb_streamlines, nheads, batch_max_len, batch_max_len]
    average_heads: bool
    average_layers: bool,
    group_with_max: bool
    lengths: List[int]
        Unpadded lengths of the streamlines.
    rescale_0_1: bool

    Returns
    -------
    attention: List[List[np.ndarray]]
        A list of nb_streamlines x
                A list of nb_layers x
                       [nheads, length, length]
        Where nheads=1 if average_heads.
    """
    if rescale_0_1:
        logging.info(
            "We will normalize the attention: per row, to the range [0, 1]: \n"
            "    The attention when deciding the next direction at point N \n"
            "    is distributed in the N first points of the streamline such\n"
            "    that the point with most attention has value 1. "
            "(att = att/max)")

    # 1. To numpy. Possibly average heads.
    nb_layers = len(attention_per_layer)
    for ll in range(nb_layers):
        # To numpy arrays
        attention_per_layer[ll] = attention_per_layer[ll].cpu().numpy()

        # Averaging heads (but keeping 4D).
        if average_heads and not group_with_max:
            attention_per_layer[ll] = np.mean(attention_per_layer[ll],
                                              axis=1, keepdims=True)

    # Possibly average layers (but keeping as list)
    if average_layers and not group_with_max:
        attention_per_layer = [np.mean(attention_per_layer, axis=0)]
        nb_layers = 1

    # 2. Rearrange attention into one list per line, unpadded, rescaled.
    attention_per_line = []
    for line in tqdm(range(len(lengths)), total=len(lengths),
                     desc="Rearranging, unpadding, rescaling (if asked)",
                     maxinterval=3):
        attention_per_line.append([None] * nb_layers)
        for layer in range(nb_layers):
            # 1. Taking one streamline, unpadding.
            line_att = attention_per_layer[layer][line, :, :, :]
            line_att = line_att[:, 0:lengths[line], 0:lengths[line]]

            # 2. Normalizing weight. Without it, we rapidly see nothing!
            # Easier to see when we normalize on the x axis.
            # Normalizing each row so that max value = 1.
            # Axis=2: Horizontally for each matrix.
            if rescale_0_1:
                line_att = line_att / np.max(line_att, axis=2, keepdims=True)

            if average_heads and group_with_max:
                line_att = np.max(line_att, axis=0, keepdims=True)

            attention_per_line[-1][layer] = line_att

        if average_layers and group_with_max:
            attention_per_line[-1] = [np.max(attention_per_line[-1], axis=0)]

    return attention_per_line


def resample_attention_one_line(line_att, this_seq_len, resample_nb):
    """
    Parameters
    ----------
    line_att: List[np.ndarray]
        A list of nb_layers x
              [nheads, length, length]
    this_seq_len: int
    resample_nb: int
    """
    if resample_nb and this_seq_len > resample_nb:
        nb_layers = len(line_att)
        nb_heads = line_att[0].shape[0]

        for ll in range(nb_layers):
            new_att = np.zeros((nb_heads, resample_nb, resample_nb))
            for h in range(nb_heads):
                ratio = resample_nb / this_seq_len
                result = zoom(line_att[ll][h, :, :], zoom=ratio,
                              order=3, mode='nearest')

                # Verifying that the future is still 0.
                result = np.tril(result)
                new_att[h, :, :] = result
            line_att[ll] = new_att
    return line_att


def prepare_decoder_tokens(this_seq_len):
    decoder_tokens = ['SOS'] + \
                     ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    return decoder_tokens


def prepare_encoder_tokens(this_seq_len, step_size, add_eos: bool):
    # If encoder = concat X | Y, then , point0 = point0 | SOS
    #                                   point1 = point1 | dir0
    # etc. But ok. We will understand.
    encoder_tokens = ['point {} ({:.1f}mm)'.format(i+1, i*step_size)
                      for i in range(this_seq_len)]

    if add_eos:
        encoder_tokens[-1] += '(SOS)'

    return encoder_tokens
