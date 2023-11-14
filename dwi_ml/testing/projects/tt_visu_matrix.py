# -*- coding: utf-8 -*-
import numpy as np
import torch

from matplotlib import pyplot as plt


def show_model_view_as_imshow(attention, fig_prefix, tokens_x, tokens_y=None):
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

    nb_layers = len(attention)
    size_x = len(tokens_x)

    if tokens_y is None:
        tokens_y = tokens_x.copy()
    size_y = len(tokens_y)

    cmap = plt.get_cmap("jet")
    cmap.set_bad(color='black')

    for i in range(nb_layers):
        att = attention[i].numpy()
        nb_heads = att.shape[1]
        fig, axs = plt.subplots(1, nb_heads, figsize=(20, 8))

        for h in range(nb_heads):
            a = np.squeeze(att[0, h, :, :])
            a = np.ma.masked_where(a == 0, a)
            im = axs[h].imshow(a, interpolation='None')
            axs[h].set_title("Head {}".format(h))
            axs[h].set_xticks(np.arange(size_x), fontsize=10)
            axs[h].set_yticks(np.arange(size_y), fontsize=10)
            axs[h].set_xticklabels(tokens_x, rotation=-90)
            axs[h].set_yticklabels(tokens_y)

            plt.colorbar(im, ax=axs[h])

        plt.suptitle("Layer {}\n"
                     "Each row: When getting next direction for this point, "
                     "what do we look at?\n"
                     "Each column: This data was used to decide which "
                     "directions?\n"
                     "DATA IS NORMALIZED TO [0-1] RANGE PER ROW".format(i))

        fig.tight_layout()
        name = fig_prefix + '_layer{}.png'.format(i)
        print("Saving matrix : {}".format(name))
        plt.savefig(name)

