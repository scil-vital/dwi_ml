# -*- coding: utf-8 -*-
import logging
from typing import Tuple

import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram


def add_attention_as_dpp(sft: StatefulTractogram, lengths,
                         attentions_per_line: Tuple, attention_name: Tuple):
    """
    Adds the attention's value to the data per point.

    Parameters
    ----------
    sft: StatefulTractogram
    lengths: list
    attentions_per_line: Tuple. For each attention:
        List of length nb_streamlines
        Such as received by unpad_rescale_attention.
        Each is: List[np.array] of length nb lines
    attention_name: Tuple[str]
    """
    assert len(attentions_per_line[0]) == len(sft), \
        ("Expecting attention to be one list per line for {} streamlines, "
         "got a list of length {}".format(len(sft), len(attentions_per_line)))
    assert isinstance(attentions_per_line[0][0], list), \
        ("Expecting attention per line to be a list (per streamline) of LIST "
         "(per layer) of attentions, but got a list of {}"
         .format(type(attentions_per_line[0][0])))
    assert isinstance(attentions_per_line[0][0][0], np.ndarray), \
        ("Expecting attention per line to be a list (per streamline) of list "
         "(per layer) of attentions AS NP.NDARRAY, but got a list of list of "
         "{}".format(type(attentions_per_line[0][0][0])))

    # Supposing the same nb layers / head for each type of attention.
    nb_layers = len(attentions_per_line[0][0])
    nb_heads = attentions_per_line[0][0][0].shape[0]

    remaining_streamlines = sft.streamlines
    whole_sft = None

    # Converting Tuple to list for easier management
    attentions_per_line = list(attentions_per_line)

    # Starting current point at length 2. At length 1, we know that it only
    # looked at the first point.
    for current_point in range(2, max(lengths)):
        # The nth point of each streamline, if long enough

        # Removing shorter streamlines from attention
        for i, att in enumerate(attentions_per_line):
            attentions_per_line[i] = \
                [a for a, s in zip(att, remaining_streamlines)
                 if len(s) > current_point]

        # Removing shorter streamlines for list of streamlines
        remaining_streamlines = [s for s in remaining_streamlines
                                 if len(s) > current_point]

        # Saving first part of streamlines, up to current_point.
        # At current_point: which point did we look at?
        # Saving many ddp key for these streamlines: per layer, per head.
        tmp_sft = sft.from_sft([s[0:current_point]
                               for s in remaining_streamlines], sft)
        logging.debug("Adding the first {} points of the remaining {} "
                      "streamlines"
                      .format(current_point+1, len(remaining_streamlines)))
        for layer in range(nb_layers):
            for head in range(nb_heads):
                for att, name in zip(attentions_per_line, attention_name):
                    # Adding data per point: attention_name_layerX_headX
                    # (Careful. Nibabel force names to be <18 character)
                    # attention_lX_hX
                    suffix = "_l{}_h{}".format(layer, head)

                    # Taking the right line of the matrix, up to the current point
                    # (i.e. before the diagonal)
                    # Nibabel required data_per_point to have the same number of
                    # dimensions as the streamlines (N, 3) = 2D. Adding a fake
                    # second dimension.
                    ddp = [a[layer][head, current_point, :current_point][:, None]
                           for a in att]
                    tmp_sft.data_per_point[name + suffix] = ddp

        if whole_sft is None:
            whole_sft = tmp_sft
        else:
            whole_sft = whole_sft + tmp_sft

    return whole_sft
