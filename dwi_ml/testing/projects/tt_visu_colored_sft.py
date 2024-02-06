# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.viz.utils import get_colormap

from dwi_ml.testing.projects.tt_visu_utils import get_visu_params_from_options, \
    prepare_colors_from_options


def _save_sft_delta(sft: StatefulTractogram, prefix: str,
                    attentions_per_line: list, attention_names: Tuple[str],
                    nb_layers, nb_heads,
                    average_heads, average_layers, group_with_max):

    for i, att_type in enumerate(attentions_per_line):
        for layer in range(nb_layers):
            if average_layers:
                if group_with_max:
                    layer_prefix = 'maxRescaledLayers'
                else:
                    layer_prefix = 'meanLayer'
            else:
                layer_prefix = 'l{}'.format(layer)

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
                    nb_points_above_thresh = np.sum(
                        weights > THRESH, axis=1, keepdims=True).astype(float)
                    dpp_nb.append(nb_points_above_thresh)

                if average_heads:
                    if group_with_max:
                        head_suffix = '_maxRescaledHeads'
                    else:
                        head_suffix = '_meanHead'
                else:
                    head_suffix = '_h{}'.format(head)

                dpp_name = 'DeltaMax'
                name = attention_names[i] + layer_prefix + head_suffix + '_'
                tractogram_name = prefix + name + dpp_name + '.trk'
                sft.data_per_point[dpp_name] = dpp_max
                sft = color_sft_from_dpp(sft, dpp_name)
                print("Saving tractogram {}".format(tractogram_name))
                save_tractogram(sft, tractogram_name)
                del sft.data_per_point[dpp_name]
                del sft.data_per_point['color']

                dpp_name = 'DeltaNb'
                tractogram_name = prefix + dpp_name + '.trk'
                sft.data_per_point[dpp_name] = dpp_nb
                sft = color_sft_from_dpp(sft, dpp_name)
                print("Saving tractogram {}".format(tractogram_name))
                save_tractogram(sft, tractogram_name)
                del sft.data_per_point[dpp_name]
                del sft.data_per_point['color']


def color_sft_duplicate_lines(
        sft: StatefulTractogram, lengths, prefix_name: str,
        attentions_per_line: list, attention_names: Tuple,
        average_heads, average_layers, group_with_max):
    """
    Saves the whole weight matrix on streamlines of all lengths.

    Output name is:
    prefix_name_colored_sft_encoder_lN_hM.trk,
                  where N is the layer, M is the head. OR:
    prefix_name_colored_sft_encoder_lN_meanH.trk
                  with option --group_heads (or _maxH). OR:
    prefix_name_colored_sft_encoder_meanL_meanH.trk
                 with option --group_all (or _maxL_maxH).
    """
    remaining_streamlines = sft.streamlines
    whole_sft = None

    # Supposing the same nb layers / head for each type of attention.
    nb_layers = len(attentions_per_line[0][0])
    nb_heads = attentions_per_line[0][0][0].shape[0]

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
                if average_layers:
                    if group_with_max:
                        layer_prefix = '_maxL'
                    else:
                        layer_prefix = '_meanL'
                else:
                    layer_prefix = 'l{}'.format(layer)

                for head in range(nb_heads):
                    # Adding data per point: attention_name_layerX_headX
                    # (Careful. Nibabel force names to be <18 character)
                    # attention_lX_hX
                    if average_heads:
                        if group_with_max:
                            head_suffix = '_maxH'
                        else:
                            head_suffix = '_meanH'
                    else:
                        head_suffix = '_h{}'.format(head)

                    # Taking the right line of the matrix, up to the current
                    # point(i.e. before the diagonal)
                    # Nibabel required data_per_point to have the same number
                    # of dimensions as the streamlines (N, 3) = 2D. Adding a
                    # fake second dimension.
                    dpp = [line_att[layer][head, current_point - 1,
                                           0:current_point][:, None]
                           for line_att in att_type]

                    name = att_name + layer_prefix + head_suffix
                    tmp_sft.data_per_point[name] = dpp

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

    # Currently, when limiting length to see the growth, smallest line stay
    # visible above the others. Tried to flip order in memory, but it does not
    # fix the view in MI-Brain. I don't know which internal order MI-Brain
    # uses.
    # order = np.flip(np.arange(len(whole_sft)))
    # whole_sft = deepcopy(whole_sft[order])

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


def color_sft_importance_where_looked(
    sft: StatefulTractogram, lengths, prefix_name: str,
    attentions_per_line: list, attention_names: Tuple,
    average_heads, average_layers,
    rescale_0_1, rescale_non_lin, rescale_z):

    (options_main, options_importance, options_where_looked,
     explanation, rescale_name) = get_visu_params_from_options(
        rescale_0_1, rescale_non_lin, rescale_z, max(lengths), max(lengths))

    # Supposing the same nb layers / head for each type of attention.
    nb_layers = len(attentions_per_line[0][0])
    nb_heads = attentions_per_line[0][0][0].shape[0]

    for i, att_type in enumerate(attentions_per_line):
        for layer in range(nb_layers):
            for head in range(nb_heads):
                colors_where_looked = []
                colors_importance = []
                for s in range(len(sft.streamlines)):
                    a = att_type[s][layer][head, :, :]
                    a, where_looked, importance = prepare_colors_from_options(
                        a, rescale_0_1, rescale_non_lin, rescale_z)
                    colors_importance.append(importance)
                    colors_where_looked.append(colors_where_looked)

                # Save results for this attention, head, layer
                name = prefix_name + '{}_l{}_h{}'.format(attention_names[i],
                                                         layer, head)
                sft.data_per_point['importance'] = colors_importance
                color_sft_from_dpp(sft, 'importance',
                                   options_importance['cmap'],
                                   options_importance['vmin'],
                                   options_importance['vmax'])
                save_tractogram(sft, name + '_importance.trk')
                del sft.data_per_point['importance']

                sft.data_per_point['where_looked'] = colors_importance
                color_sft_from_dpp(sft, 'where_looked',
                                   options_where_looked['cmap'],
                                   options_where_looked['vmin'],
                                   options_where_looked['vmax'])
                save_tractogram(sft, name + '_where_looked.trk')
                del sft.data_per_point['where_looked']


def color_sft_from_dpp(sft, key, map_name='viridis', mmin=None, mmax=None):

    cmap = get_colormap(map_name)
    tmp = [np.squeeze(sft.data_per_point[key][s]) for s in range(len(sft))]
    data = np.hstack(tmp)

    mmin = mmin or np.min(data)
    mmax = mmax or np.max(data)
    data = data - np.min(data)
    data = data / np.max(data)
    color = cmap(data)[:, 0:3] * 255
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color

    # ToDo. Save associated colorbar
    plt.figure(figsize=(1.5, 9))
    plt.imshow(np.array([[1, 0]]), cmap=cmap, vmin=mmin, vmax=mmax)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.1, 0.6, 0.85])
    plt.colorbar(orientation="vertical", cax=cax, aspect=50)

    return sft
