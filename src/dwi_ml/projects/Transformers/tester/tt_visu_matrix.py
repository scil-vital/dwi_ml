# -*- coding: utf-8 -*-
import logging

import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dwi_ml.projects.Transformers.tester.tt_visu_utils import (
    get_min_max_from_options,
    prepare_projections_from_options, get_rescale_name,
    get_explanation_projections)


def show_model_view_as_imshow(
        attention_one_line, fig_prefix, tokens_x, tokens_y,
        rescale_0_1, rescale_z, rescale_non_lin,
        average_heads, average_layers, group_with_max, cmap):
    nb_layers = len(attention_one_line)
    rescale_name = get_rescale_name(rescale_0_1, rescale_non_lin, rescale_z)
    explanation = get_explanation_projections(rescale_name)
    min_max_attention_values, min_max_position = \
        get_min_max_from_options(rescale_name)

    for i in range(nb_layers):
        layer_att = attention_one_line[i]
        nb_heads = layer_att.shape[0]
        fig, axs = plt.subplots(1, nb_heads, figsize=(20, 8),
                                layout='compressed')
        if nb_heads == 1:
            axs = [axs]

        for h in range(nb_heads):
            head_att, mean_att, importance, looked_far, max_pos, nb_looked = \
                prepare_projections_from_options(
                    layer_att[h, :, :], rescale_0_1, rescale_non_lin, rescale_z)

            divider = make_axes_locatable(axs[h])
            ax_mean_att = divider.append_axes("bottom", size=0.2, pad=0.1)
            ax_importance = divider.append_axes("bottom", size=0.2, pad=0.1)
            ax_lookedfar = divider.append_axes("right", size=0.2, pad=0)
            ax_max = divider.append_axes("right", size=0.2, pad=0)
            ax_nb_looked = divider.append_axes("right", size=0.2, pad=0)
            ax_cbar_main = divider.append_axes("right", size=0.3, pad=0.3)

            # Plot the main image
            im_main = axs[h].imshow(head_att,
                                    **min_max_attention_values, cmap=cmap,
                                    interpolation='None')

            # Bottom and right images
            _ = ax_mean_att.imshow(mean_att[None, :],
                                   **min_max_attention_values, cmap=cmap,
                                   aspect='auto', interpolation='None')
            _ = ax_importance.imshow(importance[None, :],
                                     **min_max_position, cmap=cmap,
                                     aspect='auto', interpolation='None')
            _ = ax_lookedfar.imshow(looked_far[:, None],
                                    **min_max_position, cmap=cmap,
                                    aspect='auto', interpolation='None')
            _ = ax_max.imshow(max_pos[:, None],
                              **min_max_position, cmap=cmap,
                              aspect='auto', interpolation='None')
            _ = ax_nb_looked.imshow(nb_looked[:, None],
                                    **min_max_position, cmap=cmap,
                                    aspect='auto', interpolation='None')

            # Set the titles (see also suptitle below)
            if average_heads:
                if group_with_max:
                    axs[h].set_title("Max of ({}) heads"
                                     .format(rescale_name))
                    head_suffix = "_maxOfHeads"
                else:
                    axs[h].set_title("Average of heads, {}"
                                     .format(rescale_name))
                    head_suffix = "_meanHead"
            else:
                axs[h].set_title("Head {}".format(h))
                head_suffix = "_allHeads"

            # Titles proj X
            ax_mean_att.set_ylabel("Mean", rotation=0, labelpad=25)
            ax_importance.set_ylabel("Importance.", rotation=0, labelpad=25)

            # Titles proj Y
            ax_lookedfar.set_title("Looked far", rotation=45, loc='left')
            ax_max.set_title("Max pos", rotation=45, loc='left')
            ax_nb_looked.set_title("Nb looked", rotation=45, loc='left')
            axs[h].set_xlabel("The points that the attention looks at")
            axs[h].set_ylabel("The current tractography point")

            # Move x ticks under the projections
            axs[h].tick_params(axis='x', pad=40)

            # Other plots: Hide ticks.
            for ax in [ax_mean_att, ax_importance,
                       ax_lookedfar, ax_max, ax_nb_looked]:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

            # Colorbar
            fig.colorbar(im_main, cax=ax_cbar_main)

        if average_layers:
            if group_with_max:
                layer_title = "Max of layers"
                layer_name = "_{}_maxOfLayers".format(rescale_name)
            else:
                layer_title = "Average of layers, rescaled"
                layer_name = '_meanLayer_{}'.format(rescale_name)
        else:
            layer_title = i
            layer_name = '_layer{}_{}'.format(i, rescale_name)

        plt.suptitle("Layer: {}\n{}"
                     .format(layer_title, explanation))

        name = fig_prefix + layer_name + head_suffix + '.png'
        print("Saving matrix : {}".format(layer_name + head_suffix))
        logging.info("Saved as {}".format(name))
        plt.savefig(name)
        plt.close()
