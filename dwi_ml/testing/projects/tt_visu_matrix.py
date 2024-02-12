# -*- coding: utf-8 -*-
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dwi_ml.testing.projects.tt_visu_utils import (
    get_visu_params_from_options,
    prepare_projections_from_options)


def show_model_view_as_imshow(
        attention_one_line, fig_prefix, tokens_x, tokens_y,
        rescale_0_1, rescale_z, rescale_non_lin,
        average_heads, average_layers, group_with_max):
    nb_layers = len(attention_one_line)
    size_x = len(tokens_x)
    size_y = len(tokens_y)

    (options_main, options_importance, options_range_length,
     explanation, rescale_name) = get_visu_params_from_options(
        rescale_0_1, rescale_non_lin, rescale_z, size_x, size_y)

    for i in range(nb_layers):
        att = attention_one_line[i]
        nb_heads = att.shape[0]
        fig, axs = plt.subplots(1, nb_heads, figsize=(20, 8),
                                layout='compressed')
        if nb_heads == 1:
            axs = [axs]

        for h in range(nb_heads):
            a, where_looked, importance, maxp, nb_looked = \
                prepare_projections_from_options(
                    att[h, :, :], rescale_0_1, rescale_non_lin, rescale_z)

            divider = make_axes_locatable(axs[h])
            ax_importance = divider.append_axes("bottom", size=0.2, pad=0)
            ax_lookedfar = divider.append_axes("right", size=0.2, pad=0)
            ax_max = divider.append_axes("right", size=0.2, pad=0)
            ax_nb_looked = divider.append_axes("right", size=0.2, pad=0)
            ax_cbar_main = divider.append_axes("right", size=0.3, pad=0.3)
            ax_cbar_length = divider.append_axes("right", size=0.3, pad=0.55)

            # Bottom and right images
            im_b = ax_importance.imshow(importance[None, :],
                                        **options_importance, aspect='auto')
            _ = ax_lookedfar.imshow(where_looked[:, None],
                                    **options_range_length, aspect='auto')
            _ = ax_max.imshow(maxp[:, None],
                              **options_range_length, aspect='auto')
            _ = ax_nb_looked.imshow(nb_looked[:, None],
                                    **options_range_length, aspect='auto')

            # Plot the main image
            im_main = axs[h].imshow(a, **options_main)

            # Set the titles (see also suptitle below)
            if average_heads:
                if group_with_max:
                    axs[h].set_title("Max of ({}) heads"
                                     .format(rescale_name))
                else:
                    axs[h].set_title("Average of heads, {}"
                                     .format(rescale_name))
            else:
                axs[h].set_title("Head {}".format(h))
            ax_lookedfar.set_title("Looked far", rotation=45, loc='left')
            ax_max.set_title("Max pos", rotation=45, loc='left')
            ax_nb_looked.set_title("Nb looked", rotation=45, loc='left')
            ax_importance.set_ylabel("Importance.", rotation=0, labelpad=25)
            # ("Importance" is a bit too close to last tick. Tried to use
            # loc='bottom' but then ignores labelpad).

            # Set the ticks with tokens.
            axs[h].set_xticks(np.arange(size_x), fontsize=10)
            axs[h].set_yticks(np.arange(size_y), fontsize=10)
            axs[h].tick_params(axis='x', pad=20)
            axs[h].set_xticklabels(tokens_x, rotation=-90)
            axs[h].set_yticklabels(tokens_y)

            # Other plots: Hide ticks.
            for ax in [ax_importance, ax_lookedfar, ax_max, ax_nb_looked]:
                plt.setp(ax.get_xticklabels(), visible=False)
                plt.setp(ax.get_yticklabels(), visible=False)

            # Set the colorbars, with titles.
            fig.colorbar(im_main, cax=ax_cbar_main)
            ax_cbar_main.set_ylabel('Main figure', rotation=90, labelpad=-55)
            fig.colorbar(im_b, cax=ax_cbar_length)
            ax_cbar_length.set_ylabel('x / y projections: [0, length]',
                                      rotation=90, labelpad=-55)

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

        name = fig_prefix + layer_name + '.png'
        print("Saving matrix : {}".format(name))
        plt.savefig(name)
