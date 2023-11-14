# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import numpy as np
from skimage.measure import block_reduce
import torch

from dwi_ml.models.projects.transformer_models import \
    AbstractTransformerModel


def reshape_attention(attention):
    """
    Also sends to CPU.

    Parameters
    ----------
    attention: List[Tensor]
       A list of len nb_layers with tensors:
          [nb_streamlines, batch_max_len, batch_max_len] --> If averaged heads
          [nb_streamlines, nheads, batch_max_len, batch_max_len] --> Else.

    Returns
    -------
    attention: List[Tensor]
        A list of len nb_layers with tensors:
           [nb_streamlines, nheads, batch_max_len, batch_max_len]
        Where nheads=1 if average_head.
    """
    for ll in range(len(attention)):  # Per layer
        if len(attention[ll].shape) == 3:
            # No head dimension if heads were averaged. Bertviz requires 4D.
            attention[ll] = attention[ll][:, None, :, :]
        attention[ll] = attention[ll].cpu()

    return attention


def unpad_rescale_resample_attention(
        attention, lengths, resample_nb: int = None, for_bertviz=True):
    """
    Reformats the attention to have always the same dimension, regardless of
    the model. Unpads the result. Possibly, downsample the attention for nicer
    visualisation.

    Parameters
    ----------
    attention: Tensor
        If for_bertviz: attention for ONE streamline.
    lengths: List[int]
        Unpadded lengths of the streamlines.
    resample_nb: int
        The final number of points of the attention.
    for_bertviz: bool

    Returns
    -------
    attention:
        if for_bertviz: List[List[Tensor]], else List[List[np.array]]
           List of length nb_streamline:
              Each is a list of length nb_layer:
                 Each of shape [1, nb_heads, this_s_len, this_s_len]
           (nb_heads = 1 if average_heads).
    inds: List[List[Tuple]]
    """
    if resample_nb is None:
        resample_nb = 100000

    nb_layers = len(attention)
    inds = []

    # A list of (one 4D attention per layer) per line
    attention_per_line = []
    for line in range(len(lengths)):
        this_seq_len = lengths[line]

        # 1. Verifying if we need to resample for this streamline
        ind = None
        nb_together = None
        this_resample_nb = resample_nb
        if resample_nb < this_seq_len:
            tmp = this_seq_len / resample_nb
            nb_together = np.round(tmp)
            real_resample_attention = int(np.ceil(this_seq_len / nb_together))
            nb_together = int(nb_together)

            if tmp < nb_together:
                alt = np.floor(tmp)
            else:
                alt = np.ceil(tmp)
            logging.debug(
                "NOTE: Line {}. You asked to resample the attention from {} "
                "to {}.\n"
                "      --> By combining every {} points: matrix of size {} "
                "(chosen)\n"
                "      --> By combining every {} points: matrix of size {}.\n"
                "      (We have not yet implemented an irregular resampling "
                "of the attention.) "
                .format(line, this_seq_len, resample_nb, nb_together,
                        real_resample_attention, int(alt),
                        int(np.ceil(this_seq_len / alt))))

            if nb_together > 1:
                ind1 = np.arange(0, this_seq_len, nb_together)
                ind = [(i, min(i + nb_together, this_seq_len - 1))
                       for i in ind1]
        else:
            this_resample_nb = this_seq_len

        # 2. Unpad, rescale and resample
        inds.append(ind)
        attention_per_line.append([None] * nb_layers)
        for i in range(nb_layers):
            # Easier to work with numpy. Will put back to tensor after.
            att = attention[i].numpy()
            assert len(att.shape) == 4

            # 1. Unpadding. Taking one streamline.
            att = att[line, :, 0:this_seq_len, 0:this_seq_len]

            # 2. Normalizing weight. Without it, we rapidly see nothing!
            # Easier to see when we normalize on the x axis.
            # Normalizing each row so that max value = 1.
            max_ = np.max(att, axis=2)
            att = att / max_[:, :, None]

            # 3. Resampling.
            if this_resample_nb < this_seq_len:
                # No option to pad to edge value in block_reduce. Doing manually.
                missing = (len(ind) * nb_together) - att.shape[2]
                att = np.pad(att, ((0, 0), (0, missing), (0, missing)),
                             mode='edge')

                att = block_reduce(
                    att, block_size=(1, nb_together, nb_together),
                    func=np.max, cval=1000.0)  # 1000: to see if bug.

            assert np.array_equal(att.shape[1:],
                                  [this_resample_nb, this_resample_nb]), \
                ("Hmm. Code error. Should get attention of shape "
                 "[nb_head, {}, {}], got {} (line {} layer {})"
                 .format(this_resample_nb, this_resample_nb, att.shape,
                         line, i))
            if for_bertviz:
                # Keeping torch for Bertviz
                attention_per_line[-1][i] = torch.as_tensor(att[None, :, :, :])
            else:
                attention_per_line[-1][i] = att

    return attention_per_line, inds


def prepare_decoder_tokens(streamline, ind: List[Tuple]):
    this_seq_len = len(streamline)

    if ind is not None:
        # Used resample_attention
        decoder_tokens = ['SOS-dir {}'.format(ind[0][1] - 2)] + \
                         ['dirs {} - {}'.format(i[0] - 1, i[0] - 2)
                          for i in ind]
    else:
        decoder_tokens = ['SOS'] + \
                         ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    return decoder_tokens


def prepare_encoder_tokens(this_seq_len, add_eos: bool, ind: List[Tuple]):
    # If encoder = concat X | Y, then , point0 = point0 | SOS
    #                                   point1 = point1 | dir0
    # etc. But ok. We will understand.

    if ind is not None:
        # Used resample_attention
        encoder_tokens = ['points {}-{}'.format(i[0], i[1] - 1)
                          for i in ind]
    else:
        encoder_tokens = ['point {}'.format(i) for i in range(this_seq_len)]

    if add_eos:
        encoder_tokens[-1] += '(SOS)'

    return encoder_tokens
