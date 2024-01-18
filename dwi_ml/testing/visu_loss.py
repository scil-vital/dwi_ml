# -*- coding: utf-8 -*-
"""
Runs the model, computes the loss, and saves the loss as data_per_point to view
as color.
"""
import itertools
import logging
from typing import List

import numpy as np
import torch
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt

from dwi_ml.models.main_models import ModelWithDirectionGetter

blue = [2., 75., 252.]


def prepare_colors_from_loss(
        losses: List[torch.Tensor], sft: StatefulTractogram, colormap: str,
        min_range: float = None, max_range: float = None):
    losses = np.concatenate(losses)
    # normalize between 0 and 1
    # Keeping as tensor because the split method below is easier to use
    # with torch.
    min_val = np.min(losses) if min_range is None else min_range
    max_val = np.max(losses) if max_range is None else max_range
    logging.info("Range of the colormap is considered to be: {:.2f} - {:.2f}"
                 .format(min_val, max_val))
    losses = np.clip(losses, min_val, max_val)
    losses = (losses - min_val) / (max_val - min_val)

    cmap = plt.colormaps.get_cmap(colormap)
    color = cmap(losses)[:, 0:3] * 255
    sft.data_per_point['color'] = sft.streamlines
    sft.data_per_point['color']._data = color

    colorbar_fig = plt.figure(figsize=(1.5, 9))
    plt.imshow(np.array([[1, 0]]), cmap=cmap, vmin=min_val, vmax=max_val)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.1, 0.6, 0.85])
    plt.colorbar(orientation="vertical", cax=cax, aspect=50)

    return sft, colorbar_fig


def pick_best_and_worst(nb, mean_losses):
    """
    Parameters
    ----------
    nb: int
        Will take the top n and the bottom n.
    mean_losses: list
        The lost of each streamline

    Returns
    ------
    best_idx: List
    worst_idx: List
    """
    nb_max = int(len(mean_losses) / 2)
    if nb > nb_max:
        logging.warning("You asked for the top {} and bottom {} but the "
                        "SFT contains {} streamlines; there would be overlap."
                        "Selecting the {} top and bottom."
                        .format(nb, nb, len(mean_losses), nb_max))
        nb = nb_max

    idx = np.argsort(mean_losses)
    best_idx = idx[0:nb]
    worst_idx = idx[-nb:]

    return best_idx, worst_idx


def combine_displacement_with_ref(out_dirs, sft, model):
    """
    Normalizes directions.
    Saves the model-learned streamlines together with the input streamlines:
        - This streamline is created by starting at the real position at each
        point, then advancing in the learned direction.
        - Between two points, we always add a point going back to the real
        position to view difference better. This means that the learned
        streamlines is twice the length of the real streamline.
    """
    epsilon = 0.000005
    _step_size_vox = None
    if model.step_size is not None:
        _step_size_vox = model.step_size / sft.space_attributes[2]

    out_streamlines = []
    color_x = []
    color_y = []
    color_z = []

    for i, s in enumerate(sft.streamlines):
        this_s_len = len(s)
        streamline_out_dir = out_dirs[i]

        # We expect user to have already removed the last data point if there
        # is no out_dir associated to it (if no EOS).
        assert len(streamline_out_dir) == this_s_len, \
            ("Expecting model outputs for line {} to be of length {}, got "
             "{}. Error in our code?"
             .format(i, this_s_len - 1, len(streamline_out_dir)))

        # Normalizing directions to step_size
        if _step_size_vox is not None:
            streamline_out_dir /= np.maximum(
                epsilon, np.linalg.norm(streamline_out_dir, axis=1)[:, None])
            streamline_out_dir *= _step_size_vox

        streamline_out_dir = [list(d) for d in streamline_out_dir]

        # output : Starts on first point
        #          + tmp = each point = true previous point + learned dir
        #                 + in between each point, comes back to correct point.
        tmp = [[s[p] + streamline_out_dir[p], s[p + 1]]
               for p in range(this_s_len - 1)]
        out_streamline = [s[0]] + list(itertools.chain.from_iterable(tmp))
        out_streamline = out_streamline[:-1]
        this_s_len2 = len(out_streamline)

        # Two points per point except first and last
        assert this_s_len2 == this_s_len * 2 - 2

        out_streamlines.extend([s, out_streamline])

        # Data per point: Add a color to differentiate both streamlines.
        # Ref streamline = blue
        all_x_blue = [[blue[0]]] * this_s_len
        all_y_blue = [[blue[1]]] * this_s_len
        all_z_blue = [[blue[2]]] * this_s_len
        # Learned streamline = from green to pink
        ranging_2 = [[i / this_s_len2 * 252.] for i in range(this_s_len2)]

        color_x.extend([all_x_blue, ranging_2], )
        color_y.extend([all_y_blue, [[150.]] * this_s_len2])
        color_z.extend([all_z_blue, ranging_2])

    assert len(out_streamlines) == len(sft) * 2
    data_per_point = {
        'color_x': color_x,
        'color_y': color_y,
        'color_z': color_z
    }
    sft = sft.from_sft(out_streamlines, sft, data_per_point)

    print("Tractogram displacement: \n"
          "  - In blue: The original streamline.\n"
          "  - From dark green (start) to pink: displacement of "
          "estimation at each time point.")

    return sft


def run_visu_save_colored_sft(
        losses: List[torch.Tensor], mean_losses: List, model,
        sft: StatefulTractogram,
        save_whole_tractogram, colored_sft_name: str,
        save_separate_best_and_worst, best_worst_nb,
        best_sft_name, worst_sft_name,
        colorbar_name: str, colormap: str = None,
        min_range: float = None, max_range: float = None):
    """
    Saves the losses as data per point.
    """
    assert save_whole_tractogram or save_separate_best_and_worst

    logging.info("Adding losses as data per point:")
    sft, colorbar_fig = prepare_colors_from_loss(
        losses, sft, colormap, min_range, max_range)
    print("Saving colorbar as {}".format(colorbar_name))
    colorbar_fig.savefig(colorbar_name)

    if save_whole_tractogram:
        print("Saving colored SFT as {}".format(colored_sft_name))
        save_tractogram(sft, colored_sft_name)

    if save_separate_best_and_worst:
        best_idx, worst_idx = pick_best_and_worst(best_worst_nb, mean_losses)

        best_streamlines = [sft.streamlines[i] for i in best_idx]
        worst_streamlines = [sft.streamlines[i] for i in worst_idx]
        best_sft = sft.from_sft(best_streamlines, sft)
        worst_sft = sft.from_sft(worst_streamlines, sft)
        print("Saving best and worst streamlines as {} \nand {}"
              .format(best_sft_name, worst_sft_name))
        print("Best / worst {} streamlines's losses: \n"
              "      Best : {}\n"
              "      Worst: {}"
              .format(best_worst_nb,
                      mean_losses[best_idx], mean_losses[worst_idx]))

        save_tractogram(best_sft, best_sft_name)
        save_tractogram(worst_sft, worst_sft_name)


def run_visu_save_colored_displacement(
        model: ModelWithDirectionGetter, outputs: List[torch.Tensor],
        mean_losses, sft: StatefulTractogram, displacement_sft_name: str,
        displacement_on_nb: int, displacement_on_best_and_worst: bool):
    # Select a few streamlines
    idx = []
    if displacement_on_best_and_worst:
        id_best, id_worst = pick_best_and_worst(1, mean_losses)
        idx.extend(id_best)
        idx.extend(id_worst)
    if displacement_on_nb and len(sft) > displacement_on_nb:
        idx.extend(np.random.randint(0, len(sft), size=displacement_on_nb))
    idx = np.unique(idx)
    print("Selecting {} streamlines out of {} for visualisation of the "
          "output direction.".format(len(idx), len(sft)))
    sft = sft[idx]
    if 'gaussian' in model.direction_getter.key:
        # outputs = means, sigmas
        out0 = [outputs[0][i] for i in idx]
        out1 = [outputs[1][i] for i in idx]
        outputs = (torch.vstack(out0), torch.vstack(out1))
    else:
        outputs = [outputs[i] for i in idx]
        outputs = torch.vstack(outputs)

    # Get out_dirs from model_outputs using the direction getter.
    # Use eos_thresh of 1 to be sure we don't output a NaN
    lengths = [len(s) for s in sft.streamlines]
    logging.info("Using EOS threshold 1 to avoid getting NANs. "
                 "We will get an output direction at each point even "
                 "tough the model would have rather stopped.")
    with torch.no_grad():
        out_dirs = model.get_tracking_directions(
            outputs, algo='det', eos_stopping_thresh=1.0)
        out_dirs = torch.split(out_dirs, lengths)

    out_dirs = [o.numpy() for o in out_dirs]

    # Save error together with ref
    sft = combine_displacement_with_ref(out_dirs, sft, model)

    save_tractogram(sft, displacement_sft_name, bbox_valid_check=False)
