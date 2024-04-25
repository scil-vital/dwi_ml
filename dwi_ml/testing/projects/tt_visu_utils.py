# -*- coding: utf-8 -*-
import logging
import os
from typing import List

import numpy as np
from scipy.ndimage import zoom
from tqdm import tqdm

from scilpy import get_home as get_scilpy_folder


THRESH_IMPORTANT = {
    'rescale_0_1': 0.75,  # No idea...  1 = the most important.
    'rescale_non_lin': 0.5,   # 0.5 = Value when all equal.
    'rescale_z': 1.96,  # The 95 percent confidence level with p<0.05
    'None': 0.5  # NOT SIGNIFICATIVE AT ALL.
}


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
    logging.info("Arranging attention based on visualisation options "
                 "(rescaling, averaging, etc.)")
    explanation = ''

    # 1. To numpy. Possibly average heads.
    if average_heads and not group_with_max:
        explanation += "Attentions heads on each layer have been averaged.\n"
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
        explanation += "Attention of each layer were then averaged.\n"

    # 2. Rearrange attention into one list per line, unpadded, rescaled.
    if rescale_0_1:
        explanation += ("For each streamline, attention at each point (each "
                        "row of the matrix) \nhas been "
                        "rescaled between 0-1: X = X / max(row)")
    elif rescale_z:
        explanation += ("For each streamline, attention at each point (each "
                        "row of the matrix) \nhas been "
                        "rescaled to a z-score: X = (X-mu) / std")
    elif rescale_non_lin:
        explanation += ("For each streamline, attention at each point (each "
                        "row of the matrix) \nhas been "
                        "rescaled so that 0.5 is an average point.")
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
                # Mu is not np.mean here: Ignoring future values.
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
                explanation += "We then the maximal value through heads.\n"
                line_att = np.max(line_att, axis=0, keepdims=True)

            attention_per_line[-1][layer] = line_att

        if average_layers and group_with_max:
            explanation += "We then kept the maximal value trough layers."
            attention_per_line[-1] = [np.max(attention_per_line[-1], axis=0)]

    logging.info(explanation)
    return attention_per_line, explanation


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


def get_visu_params_from_options(rescale_0_1, rescale_non_lin, rescale_z):
    """
    Defines options for prefix names, colormaps, vmin, vmax, explanation text,
    etc.
    """
    vmin_main, vmax_main, cmap_main = (0, 1, 'turbo')
    vmin_pos, vmax_pos, cmap_pos = (0, 1, 'CMRmap')
    if rescale_0_1:
        rescale_name = 'rescale_0_1'
    elif rescale_non_lin:
        rescale_name = 'rescale_non_lin'
        # cmap_main = 'coolwarm'
    elif rescale_z:
        rescale_name = 'rescale_z'
        #  Range: We could limit it to help view better. Ex: ±3 std.
        vmin_main = -3
        vmax_main = 3
    else:
        rescale_name = 'None'

    thresh = THRESH_IMPORTANT[rescale_name]
    explanation = (
        'Importance: Number of times that this point was very important '
        '(>{:.2f}).\n'
        "Looked far: Mean index of the important points (>{:.2f}) to decide "
        "the next direction. 0 = current point. 100%% = very far behind.\n"
        "Max_pos: Index of the point of maximal attention.\n"
        "Nb_looked: Number of points of important attention."
        .format(thresh, thresh))

    options_main = {'interpolation': 'None',
                    'cmap': cmap_main,
                    'vmin': vmin_main,
                    'vmax': vmax_main}

    options_position = {'interpolation': 'None',
                        'cmap': cmap_pos,
                        'vmin': vmin_pos,
                        'vmax': vmax_pos}

    return options_main, options_position, explanation, rescale_name, thresh


def prepare_projections_from_options(a, rescale_0_1, rescale_non_lin,
                                     rescale_z):
    a = np.squeeze(a)
    a = np.ma.masked_where(a == 0, a)
    if rescale_0_1:
        rescale_name = 'rescale_0_1'
    elif rescale_non_lin:
        rescale_name = 'rescale_non_lin'
    elif rescale_z:
        rescale_name = 'rescale_z'
    else:
        rescale_name = 'None'
    thresh = THRESH_IMPORTANT[rescale_name]

    length = float(a.shape[1])
    flipped_range = np.flip(np.arange(1, a.shape[1] + 1))

    # Mean = masked mean.
    mean_att = np.sum(a, axis=0) / flipped_range

    # Importance = nb of points > thresh as x projection
    importance = np.sum(a > thresh, axis=0) / length

    # Looked far = mean index of points where > thresh.
    indexes = np.arange(1, a.shape[1] + 1)
    indexes = np.abs(indexes[None, :] - indexes[:, None])
    indexes = np.ma.masked_where(~(a > thresh), indexes)
    looked_far = np.mean((a > thresh) * indexes, axis=1) / length

    # Position of maximal point
    max_pos = np.argmax(a, axis=1) + 1
    max_pos = np.arange(1, a.shape[1] + 1) - max_pos
    max_pos = max_pos / length

    # Nb looked = nb of points > thresh as y projection
    nb_looked = np.sum(a > thresh, axis=1) / length

    return a, mean_att, importance, looked_far, max_pos, nb_looked


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
