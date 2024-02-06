# -*- coding: utf-8 -*-
import numpy as np

from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from dwi_ml.testing.projects.tt_visu_utils import get_visu_params_from_options, \
    prepare_colors_from_options


def show_model_view_as_imshow(
        attention_one_line, fig_prefix, tokens_x, tokens_y,
        rescale_0_1, rescale_z, rescale_non_lin,
        average_heads, average_layers, group_with_max):
    nb_layers = len(attention_one_line)
    size_x = len(tokens_x)
    size_y = len(tokens_y)

    (options_main, options_importance, options_where_looked,
     explanation, rescale_name) = get_visu_params_from_options(
        rescale_0_1, rescale_non_lin, rescale_z)

    for i in range(nb_layers):
        att = attention_one_line[i]
        nb_heads = att.shape[0]
        fig, axs = plt.subplots(1, nb_heads, figsize=(20, 8),
                                layout='compressed')
        if nb_heads == 1:
            axs = [axs]

        for h in range(nb_heads):
            (a, where_looked, importance) = prepare_colors_from_options(
                att[h, :, :], rescale_0_1, rescale_non_lin, rescale_z)

            divider = make_axes_locatable(axs[h])
            axbottom1 = divider.append_axes("bottom", size=0.3, pad=0)
            ax_right1 = divider.append_axes("right", size=0.2, pad=0)
            ax_cbar_main = divider.append_axes("right", size=0.3, pad=0.1)
            ax_cbar_bottom1 = divider.append_axes("right", size=0.3, pad=0.4)
            ax_cbar_right1 = divider.append_axes("right", size=0.3, pad=0.4)

            # Bottom and right images
            im_bottom = axbottom1.imshow(importance[None, :],
                                         **options_importance)
            im_right = ax_right1.imshow(where_looked[:, None],
                                        **options_where_looked)

            # Plot the main image
            im_main = axs[h].imshow(a, **options_main)

            if average_heads:
                if group_with_max:
                    axs[h].set_title("Max of ({}) heads"
                                     .format(rescale_name))
                else:
                    axs[h].set_title("Average of heads, {}"
                                     .format(rescale_name))
            else:
                axs[h].set_title("Head {}".format(h))
            axs[h].set_xticks(np.arange(size_x), fontsize=10)
            axs[h].set_yticks(np.arange(size_y), fontsize=10)

            axs[h].set_xticklabels(tokens_x, rotation=-90)
            axs[h].set_yticklabels(tokens_y)
            plt.setp(axbottom1.get_xticklabels(), visible=False)
            plt.setp(axbottom1.get_yticklabels(), visible=False)
            axs[h].tick_params(axis='x', pad=20)

            fig.colorbar(im_main, cax=ax_cbar_main)
            fig.colorbar(im_bottom, cax=ax_cbar_bottom1)
            fig.colorbar(im_right, cax=ax_cbar_right1)

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
