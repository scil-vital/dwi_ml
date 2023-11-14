# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import numpy as np
from skimage.measure import block_reduce
import torch


def reshape_attention_to4d_tocpu(attention):
    """
    Also sends to CPU.

    Parameters
    ----------
    attention: List[Tensor]
       Attention such as received directly from the Transformer.
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


def unpad_rescale_attention(attention, lengths, rescale):
    """
    Reformats the attention to have always the same dimension, regardless of
    the model. Unpads the result. Possibly, downsample the attention for nicer
    visualisation.

    Parameters
    ----------
    attention: List
        Attention after running reshape_attention.
        List of len nb_layers of tensors:
             [nb_streamlines, nb_head, max_length, max_length]
    lengths: List[int]
        Unpadded lengths of the streamlines.
    rescale: bool

    Returns
    -------
    attention: List[List[np.array]]
      Length: [nb_streamlines x [nb_layers x array]]
      Arrays are of shape [nb_heads, this_s_len, this_s_len]
          (nb_heads = 1 if average_heads).
    """
    assert attention[0].shape[0] == len(lengths), \
        ("Expecting attention to be, for each layer, a tensor of shape "
         "[nb_streamlines, ...] but got shape 0={} (expected {})"
         .format(attention[0].shape[0], len(lengths)))

    if rescale:
        logging.info("We will normalize the attention: per row, to the range "
                     "[0, 1]: \nthe attention when deciding the next "
                     "direction at point N is distributed in the N first "
                     "points of the streamline such \nthat the point with "
                     "most attention has value 1. (att = att/max)")

    nb_layers = len(attention)

    # A list of (one 4D attention per layer) per line
    attention_per_line = []
    for line in range(len(lengths)):
        this_seq_len = lengths[line]

        # Unpad, rescale
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
            if rescale:
                max_ = np.max(att, axis=2)
                att = att / max_[:, :, None]

            attention_per_line[-1][i] = att

    return attention_per_line


def _verify_resampling(resample_nb, this_seq_len):
    ind = None
    this_resample_nb = resample_nb
    nb_together = None

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
            "You asked to resample the attention from {} to {}.\n"
            "      --> By combining every {} points: matrix of size {} "
            "(chosen)\n"
            "      --> By combining every {} points: matrix of size {}.\n"
            "      (We have not yet implemented an irregular resampling "
            "of the attention.) "
            .format(this_seq_len, resample_nb, nb_together,
                    real_resample_attention, int(alt),
                    int(np.ceil(this_seq_len / alt))))

        if nb_together > 1:
            ind1 = np.arange(0, this_seq_len, nb_together)
            ind = [(i, min(i + nb_together, this_seq_len - 1))
                   for i in ind1]
    else:
        this_resample_nb = this_seq_len

    return this_resample_nb, ind, nb_together


def resample_attention_one_line(att, this_seq_len,
                                resample_nb: int = None):
    """
    Parameters
    ----------
    att:
        Such as received by unpad_rescale_attention, one line only.
    this_seq_len: int
        Unpadded lengths of the streamlines.
    resample_nb: int
        The final number of points of the attention.
    """
    assert isinstance(att[0], np.ndarray), \
        ("Expecting attention to be a list, per layer, of np.ndarray, got {}"
         .format(type(att[0])))
    assert att[0].shape[1] == this_seq_len, \
        ("Expecting attention to be unpadded. For each layer, should be "
         "of shape [nb_heads, seq_len, seq_len], but got shape[1] = {} "
         "(expecting {})".format(att[0].shape[1], this_seq_len))

    if resample_nb is None:
        resample_nb = 100000

    nb_layers = len(att)

    # 1. Verifying if we need to resample for this streamline
    this_resample_nb, inds, nb_together = _verify_resampling(
        resample_nb, this_seq_len)

    # 2. Resample
    for i in range(nb_layers):
        if this_resample_nb < this_seq_len:
            # No option to pad to edge value in block_reduce. Doing manually.
            missing = (len(inds) * nb_together) - att[i].shape[2]
            att[i] = np.pad(att[i], ((0, 0), (0, missing), (0, missing)),
                            mode='edge')

            att[i] = block_reduce(
                att[i], block_size=(1, nb_together, nb_together),
                func=np.max, cval=1000.0)  # 1000: to see if bug.

        # Sending to 4D torch for Bertviz
        att[i] = torch.as_tensor(att[i])[None, :, :, :]

    return att, inds


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
