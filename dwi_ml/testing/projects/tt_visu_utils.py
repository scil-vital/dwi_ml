# -*- coding: utf-8 -*-
import os
from typing import List

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from scilpy.io.fetcher import get_home as get_scilpy_folder


def reshape_unpad_rescale_attention(
        attention_per_layer, average_heads: bool, average_layers,
        group_with_max, lengths, rescale_0_1, rescale_z, rescale_non_lin):
    """
    Get the weights as a list per layer. Transforms to a list per streamline.
    Allows unpadding.

    Also sends to CPU.

    Parameters
    ----------
    attention_per_layer: List[Tensor]
        A list: nb_layers x
                   [nb_streamlines, nheads, batch_max_len, batch_max_len]
    average_heads: bool
    average_layers: bool,
    group_with_max: bool
    lengths: List[int]
        Unpadded lengths of the streamlines.
    rescale_0_1: bool,
        If true, rescale each line of the matrix using X / max(X).
    rescale_z: bool
        If true, rescale each line of the matrix using (X-mu)/mu.
    rescale_non_lin: bool
        If true, rescale each line of the matrix to [0 - 0.5] and [0.5 - 1]

    Returns
    -------
    attention: List[List[np.ndarray]]
        A list of nb_streamlines x
                A list of nb_layers x
                       [nheads, length, length]
        Where nheads=1 if average_heads.
    """
    if rescale_0_1:
        print("Rescaling between 0-1: X = X/ max(row)")
    elif rescale_z:
        print("Rescaling using X = (X-mu) / mu.")

    # 1. To numpy. Possibly average heads.
    nb_layers = len(attention_per_layer)
    for layer in range(nb_layers):
        # To numpy arrays
        attention_per_layer[layer] = attention_per_layer[layer].cpu().numpy()

        # Averaging heads (but keeping 4D).
        if average_heads and not group_with_max:
            attention_per_layer[layer] = np.mean(attention_per_layer[layer],
                                                 axis=1, keepdims=True)

    # Possibly average layers (but keeping as list)
    if average_layers and not group_with_max:
        attention_per_layer = [np.mean(attention_per_layer, axis=0)]
        nb_layers = 1

    # 2. Rearrange attention into one list per line, unpadded, rescaled.
    attention_per_line = []
    for line in tqdm(range(len(lengths)), total=len(lengths),
                     desc="Rearranging, unpadding, rescaling (if asked)",
                     maxinterval=3):
        attention_per_line.append([None] * nb_layers)
        for layer in range(nb_layers):
            # 1. Taking one streamline, unpadding.
            line_att = attention_per_layer[layer][line, :, :, :]
            line_att = line_att[:, 0:lengths[line], 0:lengths[line]]

            # 2. Normalizing weight. Without it, we rapidly see nothing!
            # Easier to see when we normalize on the x axis.
            # Normalizing each row so that max value = 1.
            # Axis=2: Horizontally for each matrix.
            if rescale_0_1:
                # Rescale [0, max] -->  [0, 1]
                line_att = line_att / np.max(line_att, axis=2, keepdims=True)
            elif rescale_z:
                # Mu is not np.mean here. Ignoring future values.
                # Expected values for mu = [1, 1/2, 1/3, etc.]
                mask = np.ones((line_att.shape[1], line_att.shape[2]))
                mask = np.tril(mask)[None, :, :]

                nb = np.arange(line_att.shape[1]) + 1

                mu = np.sum(line_att, axis=2) / nb[None, :]
                mu = mu[:, :, None]

                std = (line_att - mu) ** 2
                std = np.sqrt(np.sum(std * mask, axis=2) / nb[None, :])
                std = std[:, :, None]

                line_att = (line_att - mu) / np.maximum(std, 1e-6)

                # Back to triu
                line_att = line_att * mask
            elif rescale_non_lin:
                nb = np.arange(line_att.shape[1]) + 1
                mean = 1 / nb[None, :, None]

                where_below = line_att <= mean
                where_above = ~where_below

                # Rescale [0, mean] --> [0, 0.5]
                #   x = (  (x-0) / (mean - 0)    )*0.5
                tmp1 = 0.5 * (line_att / mean)

                # Rescale [mean, 1] --> [0.5, 1]
                #   x = 0.5 +   (    (x - mean) /  (1-mean)    ) * 0.5
                # But (1 - mean) creates an error for first point. It does not
                # belong to where_above so we don't care about this value.
                mean[:, 0, :] = 10
                tmp2 = 0.5 + 0.5 * ((line_att - mean) / (1.0 - mean))

                line_att = tmp1 * where_below + tmp2 * where_above

            if average_heads and group_with_max:
                line_att = np.max(line_att, axis=0, keepdims=True)

            attention_per_line[-1][layer] = line_att

        if average_layers and group_with_max:
            attention_per_line[-1] = [np.max(attention_per_line[-1], axis=0)]

    return attention_per_line


def resample_attention_one_line(line_att, this_seq_len, resample_nb):
    """
    Parameters
    ----------
    line_att: List[np.ndarray]
        A list of nb_layers x
              [nheads, length, length]
    this_seq_len: int
    resample_nb: int
    """
    if resample_nb and this_seq_len > resample_nb:
        nb_layers = len(line_att)
        nb_heads = line_att[0].shape[0]

        for ll in range(nb_layers):
            new_att = np.zeros((nb_heads, resample_nb, resample_nb))
            for h in range(nb_heads):
                ratio = resample_nb / this_seq_len
                result = zoom(line_att[ll][h, :, :], zoom=ratio,
                              order=3, mode='nearest')

                # Verifying that the future is still 0.
                result = np.tril(result)
                new_att[h, :, :] = result
            line_att[ll] = new_att
    return line_att


def prepare_decoder_tokens(this_seq_len):
    decoder_tokens = ['SOS'] + \
                     ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    return decoder_tokens


def prepare_encoder_tokens(this_seq_len, step_size, add_eos: bool):
    # If encoder = concat X | Y, then , point0 = point0 | SOS
    #                                   point1 = point1 | dir0
    # etc. But ok. We will understand.
    encoder_tokens = ['p {} ({:.1f}mm)'.format(i + 1, i * step_size)
                      for i in range(this_seq_len)]

    if add_eos:
        encoder_tokens[-1] += '(SOS)'

    return encoder_tokens


def get_visu_params_from_options(
        rescale_0_1, rescale_non_lin, rescale_z, size_x, size_y):
    """
    Defines options for prefix names, colormaps, vmin, vmax, explanation text,
    etc.
    """
    vmin_main, vmax_main, cmap_main = (0, 1, 'viridis')
    vmin_bottom, vmax_bottom, cmap_bottom = (0, 1, 'viridis')
    vmin_right, vmax_right, cmap_right = (0, 1, 'viridis')
    if rescale_0_1 or rescale_non_lin:
        if rescale_0_1:
            explanation = (
                'Data is rescaled: to [0-1] range per row.\n'
                'Bottom row = Importance: Number of times that this point was '
                'very important (> 0.9).\n'
                "Right column = Where looked: Where the important points "
                "(>0.9) to decide next direction are situated.\n"
                "0 = looks at current point. Max = looks very far behind.\n"
                'cbar1: main matrix. cbar2: bottom row. cbar3: right column')
            rescale_name = 'rescale_0_1'
        else:
            explanation = (
                "Data is rescaled: On each row, a value of 0.5 means "
                "that the point had an average importance (raw value "
                "was 1/N)\n"
                "Bottom row = Importance: Number of times that this point was "
                "more important than the average >0.5.\n"
                "Right column = Where looked: Where the important points "
                "(>0.5) to decide next direction are situated.\n"
                "0 = looks at current point. Max = looks very far behind.\n"
                "cbar1: main matrix. cbar2: bottom row. cbar3: right column")
            rescale_name = 'rescale_non_lin'
            cmap_main = 'rainbow'  # See also turbo
        vmax_bottom = size_y
        cmap_bottom = 'plasma'  # See also rainbow, inferno
        vmax_right = size_x
        cmap_right = 'plasma'
    elif rescale_z:
        explanation = ("Data is rescaled to z-scores per row.\n"
                       "Bottom row: Average.")
        rescale_name = 'rescale_z'
        vmin_main = None
        vmax_main = None
        vmax_bottom = None
        vmin_bottom = None
    else:
        explanation = 'Bottom row: Average.'
        rescale_name = ''

    options_main = {'interpolation': 'None',
                    'cmap': cmap_main,
                    'vmin': vmin_main,
                    'vmax': vmax_main}

    options_importance = {'interpolation': 'None',
                          'cmap': cmap_bottom,
                          'vmin': vmin_bottom,
                          'vmax': vmax_bottom}

    options_where_looked = {'interpolation': 'None',
                            'cmap': cmap_right,
                            'vmin': vmin_right,
                            'vmax': vmax_right}

    return (options_main, options_importance, options_where_looked,
            explanation, rescale_name)


def prepare_colors_from_options(a, rescale_0_1, rescale_non_lin, rescale_z):
    a = np.squeeze(a)
    a = np.ma.masked_where(a == 0, a)
    if rescale_0_1 or rescale_non_lin:
        if rescale_0_1:
            thresh = 0.9
        else:
            thresh = 0.5

        # To set as percent: / np.flip(np.arange(1, a.shape[1] + 1))
        importance = np.sum(a > thresh, axis=0)
        indexes = np.arange(1, a.shape[1] + 1)
        indexes = np.abs(indexes[None, :] - indexes[:, None])
        indexes = np.ma.masked_where(~(a > thresh), indexes)
        where_looked = np.mean((a > thresh) * indexes, axis=1)
    else:
        importance = np.mean(a, axis=0)
        raise NotImplemented("Where looked not defined.")

    return a, where_looked, importance


def get_config_filename():
    """
    File that will be saved by the python script with all the args. The
    jupyter notebook can then load them again.
    """
    # We choose to add it in the hidden .scilpy folder in our home.
    # (Where our test data also is).
    hidden_folder = get_scilpy_folder()
    config_filename = os.path.join(
        hidden_folder, 'ipynb_tt_visualize_weights.config')
    return config_filename


def get_out_dir_and_create(args):
    # Define out_dir as experiment_path/visu_weights if not defined.
    # Create it if it does not exist.
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu_weights')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    return args
