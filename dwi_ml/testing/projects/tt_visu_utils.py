# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import numpy as np
import torch
from scipy.ndimage import zoom
from tqdm import tqdm


def reshape_unpad_rescale_attention(attention_per_layer, average_heads: bool,
                                    lengths, rescale, resample_nb):
    """
    Also sends to CPU.

    Parameters
    ----------
    attention_per_layer: List[Tensor]
       Attention such as received directly from the Transformer.
       A list of len nb_layers with tensors:
          [nb_streamlines, batch_max_len, batch_max_len] --> If averaged heads
          [nb_streamlines, nheads, batch_max_len, batch_max_len] --> Else.
    average_heads: bool
        If true, average heads.
    lengths: List[int]
        Unpadded lengths of the streamlines.
    rescale: bool
    resample_nb: int or None

    Returns
    -------
    attention: List[np.ndarray]
        A list of len nb_streamlines with, each:
            A list of len nb_layers of np.ndarray of shape:
                [nheads, batch_max_len, batch_max_len]
        Where nheads=1 if average_head.
    """
    nb_layers = len(attention_per_layer)
    nb_streamlines = len(lengths)

    if rescale:
        logging.info("We will normalize the attention: per row, to the range "
                     "[0, 1]: \n"
                     "    The attention when deciding the next direction at "
                     "point N is \n"
                     "    distributed in the N first points of the streamline "
                     "such that \n"
                     "    the point with most attention has value 1. "
                     "(att = att/max)")

    # 1. Rearrange attention per layer to 4D
    for ll in range(nb_layers):
        # To numpy arrays
        attention_per_layer[ll] = attention_per_layer[ll].cpu().numpy()

        # Averaging heads (but keeping 4D).
        if average_heads:
            attention_per_layer[ll] = np.mean(attention_per_layer[ll],
                                              axis=1, keepdims=True)

        assert attention_per_layer[ll].shape[0] == nb_streamlines, \
            ("Expecting attention to be, for each layer, a tensor of shape "
             "[nb_streamlines={}, nb_heads, max_len, max_len] but got "
             "shape[0] = {}."
             .format(len(lengths), attention_per_layer[0].shape[0]))

    nb_heads = attention_per_layer[0].shape[1]

    # 2. Rearrange attention into one list per line, unpadded, rescaled.
    attention_per_line = []
    for line in tqdm(range(len(lengths)), total=len(lengths),
                     desc="Rearranging, unpadding, rescaling (if asked)",
                     maxinterval=3):
        this_seq_len = lengths[line]

        attention_per_line.append([None] * nb_layers)
        for i in range(nb_layers):
            # 1. Unpadding. Taking one streamline.
            att = attention_per_layer[i][line, :, 0:this_seq_len, 0:this_seq_len]

            # 2. Resampling
            if resample_nb and this_seq_len > resample_nb:
                new_att = np.zeros((nb_heads, resample_nb, resample_nb))
                for h in range(nb_heads):
                    ratio = resample_nb / this_seq_len
                    result = zoom(att[h, :, :], zoom=ratio,
                                  order=3, mode='nearest')

                    # Verifying that the future is still 0.
                    result = np.tril(result)
                    new_att[h, :, :] = result

            # 3. Normalizing weight. Without it, we rapidly see nothing!
            # Easier to see when we normalize on the x axis.
            # Normalizing each row so that max value = 1.
            if rescale:
                max_ = np.max(att, axis=2)
                att = att / max_[:, :, None]

            attention_per_line[-1][i] = att

    return attention_per_line


def prepare_decoder_tokens(this_seq_len, ind: List[Tuple]):

    if ind is not None:
        # Used resample_attention
        decoder_tokens = ['SOS-dir {}'.format(ind[0][1] - 2)] + \
                         ['dirs {}-{}'.format(i[0] - 1, i[1] - 2)
                          for i in ind[1:]]
    else:
        decoder_tokens = ['SOS'] + \
                         ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    return decoder_tokens


def prepare_encoder_tokens(this_seq_len, add_eos: bool):
    # If encoder = concat X | Y, then , point0 = point0 | SOS
    #                                   point1 = point1 | dir0
    # etc. But ok. We will understand.

    encoder_tokens = ['point {}'.format(i) for i in range(this_seq_len)]

    if add_eos:
        encoder_tokens[-1] += '(SOS)'

    return encoder_tokens
