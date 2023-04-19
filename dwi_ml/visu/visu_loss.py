# -*- coding: utf-8 -*-
"""
Runs the model, computes the loss, and saves the loss as data_per_point to view
as color.
"""
import itertools
from argparse import ArgumentParser
import logging
from typing import List

import numpy as np
import torch
from matplotlib import pyplot as plt
from scilpy.io.utils import add_reference_arg, add_overwrite_arg

from dwi_ml.io_utils import add_logging_arg, add_arg_existing_experiment_path
from dwi_ml.io_utils import add_memory_args
from dwi_ml.testing.utils import add_args_testing_subj_hdf5

blue = [2., 75., 252.]


def prepare_args_visu_loss(p: ArgumentParser, use_existing_experiment=True):
    # Mandatory
    if use_existing_experiment:
        # Should only be False for debugging tests.
        add_arg_existing_experiment_path(p)

    add_args_testing_subj_hdf5(p)

    # Options
    p.add_argument('--batch_size', type=int)
    add_memory_args(p)

    g = p.add_argument_group("Options to save loss as a colored SFT")
    g.add_argument('--save_colored_tractogram', metavar='out_name.trk',
                   dest='out_colored_sft',
                   help="If set, saves the tractogram with the loss per point "
                        "as a data per point (color)")
    g.add_argument('--min_range', type=float,
                   help="Inferior range of the colormap. If any loss is lower"
                        "than that value, they will be clipped.")
    g.add_argument('--max_range', type=float)
    g.add_argument('--colormap', default='plasma',
                   help="Select the colormap for colored trk [%(default)s].\n"
                        "Can be any matplotlib cmap.")
    g.add_argument('--save_best_and_worst', type=int, nargs='?', const=10,
                   metavar='x',
                   help="Save separately the worst x%% and best x%% of "
                        "streamlines. Default: 10%%.")
    g.add_argument('--show_colorbar')

    g = p.add_argument_group("Options to save output direction as "
                             "displacement")
    g.add_argument('--save_displacement', metavar='out_name.trk',
                   dest='out_displacement_sft',
                   help="If set, picks one streamline and computes the "
                        "outputs at each position.\n Saves a streamline that "
                        "starts at each real coordinates and moves in the "
                        "output direction.")
    g.add_argument('--pick_at_random', action='store_true')
    g.add_argument('--pick_best_and_worst', action='store_true')
    g.add_argument('--pick_idx', type=int, nargs='*')

    add_overwrite_arg(p)
    add_logging_arg(p)


def prepare_colors_from_loss(
        losses: torch.Tensor, contains_eos, sft, colormap,
        min_range=None, max_range=None, skip_first_point=False,
        nan_value=None):
    """
    Args
    ----
    losses: list[Tensor]
    contains_eos: bool
        If true, the EOS loss is added at the last point. Else, the nan_value
        is added.
    sft: StatefulTractogram
    colormap: str
    min_range: float
    max_range: float
    skip_first_point: bool
        If true, nan_value is used as loss at the first coordinate.
        Meant particularly for the "copy_previous_dir" loss, where there is no
        previous dir at the first point.
    nan_value: float
        Value to use when loss is unknown. If not given, will be set to the
        minimal value.
    """
    # normalize between 0 and 1
    # Keeping as tensor because the split method below is easier to use
    # with torch.
    min_val = torch.min(losses) if min_range is None else min_range
    max_val = torch.max(losses) if max_range is None else max_range
    logging.info("Range of the colormap is considered to be: {:.2f} - {:.2f}"
                 .format(min_val, max_val))
    losses = torch.clip(losses, min_val, max_val)
    losses = (losses - min_val) / (max_val - min_val)
    nan_value = nan_value if nan_value is not None else 0.0
    print("Loss ranges between {} and {}".format(min_val, max_val))

    # Splitting back into streamlines to add a 0 loss at last position
    # (if no EOS was used).
    diff_length = 0 if contains_eos else 1
    if skip_first_point:
        diff_length += 1
    lengths = [len(s) - diff_length for s in sft.streamlines]
    losses = torch.split(losses, lengths)
    losses = [s_losses.tolist() for s_losses in losses]

    # Add 0 loss for the last point and merge back
    if skip_first_point:
        logging.info("Loss unkown at first point of the streamlines. Set to {}"
                     .format(nan_value))
        losses = [[nan_value] + s_losses for s_losses in losses]

    if not contains_eos:
        logging.info("Loss unkown at last point of the streamlines. Set to {}"
                     .format(nan_value))
        losses = [s_losses + [nan_value] for s_losses in losses]

    losses = np.concatenate(losses)

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


def separate_best_and_worst(percent, contains_eos, losses, sft):
    diff_length = 0 if contains_eos else 1
    lengths = [len(s) - diff_length for s in sft.streamlines]
    losses = torch.split(losses, lengths)
    losses = [np.mean(s_losses) for s_losses in losses]

    percent = int(percent / 100 * len(sft))
    idx = np.argsort(losses)
    best_idx = idx[0:percent]
    worst_idx = idx[-percent:]

    print("Best / worst streamline's loss: \n"
          "      Worst : {}\n"
          "      Best: {}".format(losses[best_idx[0]], losses[worst_idx[-1]]))


def pick_a_few(sft, ids_best, ids_worst,
               pick_at_random: bool, pick_best_and_worst: bool,
               pick_idx: List[int]):
    chosen_streamlines = []
    if pick_at_random:
        chosen_streamlines.extend(np.random.randint(0, len(sft), size=1))
    if pick_best_and_worst:
        chosen_streamlines.extend(ids_best[0])
        chosen_streamlines.extend(ids_worst[-1])
    if pick_idx is not None and len(pick_idx) > 0:
        chosen_streamlines.extend(pick_idx)

    chosen_streamlines = np.unique(chosen_streamlines)
    return sft[chosen_streamlines], chosen_streamlines


def combine_displacement_with_ref(out_dirs, sft, step_size_mm=None):
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
    if step_size_mm is not None:
        _step_size_vox = step_size_mm / sft.space_attributes[2]

    out_streamlines = []
    color_x = []
    color_y = []
    color_z = []
    for i, s in enumerate(sft.streamlines):
        this_s_len = len(s)

        # Normalizing directions to step_size
        streamline_out_dir = out_dirs[i]

        if _step_size_vox is not None:
            streamline_out_dir /= np.maximum(
                epsilon, np.linalg.norm(streamline_out_dir, axis=1)[:, None])
            streamline_out_dir *= _step_size_vox

        streamline_out_dir = [list(d) for d in streamline_out_dir]
        assert len(streamline_out_dir) == this_s_len - 1

        # output : Starts on first point
        #          + tmp = each point = true previous point + learned dir
        #                 + in between each point, comes back to correct point.
        logging.warning("S: {}\n D: {}\n".format(s, streamline_out_dir))
        tmp = [[s[p] + streamline_out_dir[p], s[p+1]]
               for p in range(this_s_len - 1)]
        out_streamline = \
            [s[0]] + list(itertools.chain.from_iterable(tmp))
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

        color_x.extend([all_x_blue, ranging_2],)
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
