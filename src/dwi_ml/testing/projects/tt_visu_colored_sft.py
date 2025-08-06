# -*- coding: utf-8 -*-
from copy import deepcopy
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram

from scilpy.viz.color import get_lookup_table

from dwi_ml.testing.projects.tt_visu_utils import (
    get_visu_params_from_options,
    prepare_projections_from_options)


def color_sft_duplicate_lines(
        sft: StatefulTractogram, lengths, prefix_name: str,
        attentions_per_line: list, attention_names: Tuple,
        average_heads, average_layers, group_with_max, explanation):
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

    # Avoid duplicating the attention in-place.
    attentions_per_line = deepcopy(attentions_per_line)

    # Supposing the same nb layers / head for each type of attention.
    nb_layers = len(attentions_per_line[0][0])
    nb_heads = attentions_per_line[0][0][0].shape[0]

    # Duplicating!
    # Using s[0:current_point], so starting at current point 2, to have
    # [s0, s1]. Else, cannot visualize a single point.
    # (Anyway at point 0: Always looking at point 0 only)
    remaining_streamlines = sft.streamlines
    whole_sft = None
    for current_point in range(2, max(lengths) + 1):

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
        # We will separate them later at saving time.
        # (Careful. Nibabel force names to be <18 character)
        # encoder_lX_hX
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

                    dpp_name = att_name + layer_prefix + head_suffix
                    tmp_sft.data_per_point[dpp_name] = dpp

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
        # Keep only current key
        colored_sft = whole_sft.from_sft(
            whole_sft.streamlines, whole_sft,
            data_per_point={key: whole_sft.data_per_point[key]})

        # Not using a fixed vmax, vmin. Uses the bundle's data.
        # Easier to view.
        colored_sft, cbar_fig = _color_sft_from_dpp(
            colored_sft, key, prepare_fig=True, title=explanation)

        filename_prefix = prefix_name + '_colored_multi_length_' + key
        filename_trk = filename_prefix + '.trk'
        filename_cbar = filename_prefix + '_cbar.png'
        print("Saving {} with dpp: {}"
              .format(filename_trk, list(colored_sft.data_per_point.keys())))

        save_tractogram(colored_sft, filename_trk, bbox_valid_check=False)
        plt.savefig(filename_cbar)


def color_sft_x_y_projections(
        sft: StatefulTractogram, prefix_name: str,
        attentions_per_line: list, attention_names: Tuple,
        average_heads, average_layers, group_with_max,
        rescale_0_1, rescale_non_lin, rescale_z, explanation):
    """
    Saves one tractogram per "projection":

    Proj on x:
        - nb_usage
        - mean_att

    Proj on y:
        - looked_far
        - max_pos
        - nb_looked
    """
    (options_main, options_range_length, explanation_part2,
     rescale_name, thresh) = get_visu_params_from_options(
        rescale_0_1, rescale_non_lin, rescale_z)

    explanation += '\n' + explanation_part2

    # Supposing the same nb layers / head for each type of attention.
    nb_layers = len(attentions_per_line[0][0])
    nb_heads = attentions_per_line[0][0][0].shape[0]

    for i, att_type in enumerate(attentions_per_line):
        for layer in range(nb_layers):
            if average_layers:
                if group_with_max:
                    layer_prefix = '_maxL'
                else:
                    layer_prefix = '_meanL'
            else:
                layer_prefix = 'l{}'.format(layer)
            for head in range(nb_heads):
                if average_heads:
                    if group_with_max:
                        head_suffix = '_maxH'
                    else:
                        head_suffix = '_meanH'
                else:
                    head_suffix = '_h{}'.format(head)

                all_nb_usage = []
                all_mean_att = []
                all_looked_far = []
                all_maxp = []
                all_nb_looked = []
                for s in range(len(sft.streamlines)):
                    a = att_type[s][layer][head, :, :]
                    a, mean_att, nb_usage, looked_far, max_p, nb_looked = \
                        prepare_projections_from_options(
                            a, rescale_0_1, rescale_non_lin, rescale_z)
                    all_mean_att.append(mean_att[:, None])
                    all_nb_usage.append(nb_usage[:, None])
                    all_looked_far.append(looked_far[:, None])
                    all_maxp.append(max_p[:, None])
                    all_nb_looked.append(nb_looked[:, None])

                # Save results for this attention, head, layer
                filename_prefix = prefix_name + attention_names[i] + \
                    layer_prefix + head_suffix

                # Mean att: not fixing the vmin, vmax.
                name = 'x_mean_att'
                sft.data_per_point[name] = all_mean_att
                _color_sft_from_dpp(sft, name, **options_range_length)
                filename_trk = filename_prefix + '_' + name + '.trk'
                print("Saving {} with dpp {}"
                      .format(filename_trk, list(sft.data_per_point.keys())))
                save_tractogram(sft, filename_trk, bbox_valid_check=False)
                del sft.data_per_point[name]
                del sft.data_per_point['color']

                # Others: Range is 0 - 1 (where 1 = 100% of streamline length)
                # Not saving the colorbar.
                names = ('x_nb_usage',
                         'y_looked_far', 'y_max_pos', 'y_nb_looked')
                data = (all_nb_usage,
                        all_looked_far, all_maxp, all_nb_looked)
                for name, vectors in zip(names, data):
                    sft.data_per_point[name] = vectors
                    _color_sft_from_dpp(sft, name, **options_range_length)
                    filename_trk = filename_prefix + '_' + name + '.trk'
                    print("Saving {} with dpp {}"
                          .format(filename_trk,
                                  list(sft.data_per_point.keys())))
                    save_tractogram(sft, filename_trk, bbox_valid_check=False)
                    del sft.data_per_point[name]
                    del sft.data_per_point['color']


def _color_sft_from_dpp(sft, key, cmap='viridis', vmin=None, vmax=None,
                        prepare_fig: bool = False, title=None, **kw):
    cmap = get_lookup_table(cmap)
    tmp = [np.squeeze(sft.data_per_point[key][s]) for s in range(len(sft))]
    data = np.hstack(tmp)

    vmin = vmin or np.min(data)
    vmax = vmax or np.max(data)
    data = data - np.min(data)
    data = data / np.max(data)
    color = cmap(data)[:, 0:3] * 255
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color

    # Preparing a figure
    fig = None
    if prepare_fig:
        fig = plt.figure(figsize=(9, 3))
        plt.imshow(np.array([[1, 0]]), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.gca().set_visible(False)
        cax = plt.axes([0.1, 0.1, 0.6, 0.85])
        plt.colorbar(orientation="horizontal", cax=cax, aspect=0.01)
        if title is not None:
            plt.title("Colorbar for key: {}\n".format(key) + title)
        else:
            plt.title("Colorbar for key: {}".format(key))

    return sft, fig
