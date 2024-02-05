# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from scilpy.viz.utils import get_colormap


def save_sft_with_attention_as_dpp(
        sft: StatefulTractogram, lengths, prefix_name: str,
        attentions_per_line: Tuple, attention_names: Tuple,
        sft_delta: bool = False, sft_all_info: bool = True):
    """
    Adds the attention's value to the data per point.

    Parameters
    ----------
    sft: StatefulTractogram
    lengths: list
    prefix_name: str
    attentions_per_line: Tuple. For each attention:
        List of length nb_streamlines (after unpad_rescale_attention).
        Each is: List[np.array] of length nb layers.
        Each attention is of shape: [nb_heads, line_length, line_length]
    attention_names: Tuple[str]
    sft_delta: bool
        If true, save the "Delta" colors: save a file '_max' and a file '_nb'.
    sft_all_info: bool
        If true, save the streamlines of all lengths with color at each point.
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

    # Converting Tuple to list for easier management
    attentions_per_line = list(attentions_per_line)

    if sft_delta:
        _save_sft_delta(sft, prefix_name, attentions_per_line,
                        attention_names, nb_layers, nb_heads)

    if sft_all_info:
        _save_sft_all_info(sft, lengths, prefix_name, attentions_per_line,
                           attention_names, nb_layers, nb_heads)


def _save_sft_delta(sft: StatefulTractogram, prefix_name: str,
                    attentions_per_line: list, attention_names: Tuple[str],
                    nb_layers, nb_heads):

    for i, att_type in enumerate(attentions_per_line):
        for layer in range(nb_layers):
            for head in range(nb_heads):
                dpp_max = []
                dpp_nb = []
                for s in range(len(sft.streamlines)):
                    weights = att_type[s][layer][head, :, :]
                    max_ = np.argmax(weights, axis=1,
                                     keepdims=True).astype(float)

                    # Currently max_ is the index of the point with maximal
                    # attention. Changing to be a ratio of the streamline
                    # length, independent of the length.
                    max_ /= len(sft.streamlines[s])
                    dpp_max.append(max_)

                    # Mean, mean weighted, etc.: Do not seem to represent much.
                    THRESH = 0.5
                    nb_points_above_thresh = np.sum(weights > THRESH, axis=1,
                                                    keepdims=True).astype(float)
                    dpp_nb.append(nb_points_above_thresh)

                prefix = attention_names[i] + 'l{}_h{}'.format(layer, head)

                dpp = prefix + '_max'
                tractogram_name = prefix_name + dpp + '.trk'
                sft.data_per_point[dpp] = dpp_max
                sft = color_sft_from_dpp(sft, dpp)
                print("Saving tractogram {}".format(tractogram_name))
                save_tractogram(sft, tractogram_name)
                del sft.data_per_point[dpp]
                del sft.data_per_point['color']

                dpp = prefix + '_nb'
                tractogram_name = prefix_name + dpp + '.trk'
                sft.data_per_point[dpp] = dpp_nb
                sft = color_sft_from_dpp(sft, dpp)
                print("Saving tractogram {}".format(tractogram_name))
                save_tractogram(sft, tractogram_name)
                del sft.data_per_point[dpp]
                del sft.data_per_point['color']


def _save_sft_all_info(sft: StatefulTractogram, lengths, prefix_name: str,
                       attentions_per_line: list, attention_names: Tuple,
                       nb_layers, nb_heads):
    remaining_streamlines = sft.streamlines
    whole_sft = None

    # Using s[0:current_point], so starting current point 2, to have
    # [s0, s1]. Else, cannot visualize a single point.
    # (Anyway at point 0: Always looking at point 0 only)
    for current_point in range(2, max(lengths) + 1):
        # The nth point of each streamline, if long enough

        for i, att_type in enumerate(attentions_per_line):
            # Removing shorter streamlines from each type of attention
            # (encoder, decoder, cross)
            attentions_per_line[i] = [line_att for line_att, s in
                                      zip(att_type, remaining_streamlines)
                                      if len(s) >= current_point]

        # Removing shorter streamlines for list of streamlines
        remaining_streamlines = [s for s in remaining_streamlines
                                 if len(s) >= current_point]

        # Saving first part of streamlines, up to current_point:
        #  = "At current_point: which point did we look at?"
        tmp_sft = sft.from_sft([s[0:current_point]
                               for s in remaining_streamlines], sft)

        # Saving many ddp key for these streamlines: per layer, per head.
        for att_name, att_type in zip(attention_names, attentions_per_line):
            for layer in range(nb_layers):
                for head in range(nb_heads):
                    # Adding data per point: attention_name_layerX_headX
                    # (Careful. Nibabel force names to be <18 character)
                    # attention_lX_hX
                    suffix = "_l{}_h{}".format(layer, head)

                    # Taking the right line of the matrix, up to the current point
                    # (i.e. before the diagonal)
                    # Nibabel required data_per_point to have the same number of
                    # dimensions as the streamlines (N, 3) = 2D. Adding a fake
                    # second dimension.
                    dpp = [line_att[layer][head, current_point - 1,
                                           0:current_point][:, None]
                           for line_att in att_type]

                    tmp_sft.data_per_point[att_name + suffix] = dpp

        if whole_sft is None:
            whole_sft = tmp_sft
        else:
            whole_sft = whole_sft + tmp_sft

    print("  **The initial {} streamlines were transformed into {} "
          "streamlines of \n"
          "     variable lengths. Color for streamline i of length N is the "
          "attention's value \n"
          "     at each point when deciding the next direction at point N."
          .format(len(sft), len(whole_sft.streamlines)))
    del sft
    del tmp_sft

    # Inverting the order. Right now: smallest to biggest. But in Mi-Brain,
    # The first has priority on the view. So when limiting length to see the
    # growth, smallest line is always visible above the others.
    order = np.flip(np.arange(len(whole_sft)))
    whole_sft = deepcopy(whole_sft[order])

    dpp_keys = list(whole_sft.data_per_point.keys())
    for key in dpp_keys:
        name = prefix_name + '_colored_sft_' + key + '.trk'
        # Keep only current key
        colored_sft = whole_sft.from_sft(
            whole_sft.streamlines, whole_sft,
            data_per_point={key: whole_sft.data_per_point[key]})
        colored_sft = color_sft_from_dpp(colored_sft, key)

        print("Saving {} with dpp: {}"
              .format(name, list(colored_sft.data_per_point.keys())))

        save_tractogram(colored_sft, name)


def color_sft_from_dpp(sft, key):

    cmap = get_colormap('viridis')
    tmp = [np.squeeze(sft.data_per_point[key][s]) for s in range(len(sft))]
    data = np.hstack(tmp)

    mmin = np.min(data)
    mmax = np.max(data)
    data = data - np.min(data)
    data = data / np.max(data)
    color = cmap(data)[:, 0:3] * 255
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color

    DEBUG = False
    if DEBUG:
        plt.figure(figsize=(1.5, 9))
        plt.imshow(np.array([[1, 0]]), cmap=cmap, vmin=mmin, vmax=mmax)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.1, 0.6, 0.85])
        plt.colorbar(orientation="vertical", cax=cax, aspect=50)
        plt.show()

    return sft
