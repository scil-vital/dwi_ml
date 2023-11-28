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
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt
from scilpy.io.utils import add_overwrite_arg

from dwi_ml.io_utils import add_arg_existing_experiment_path, add_logging_arg, add_memory_args
from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.testing.utils import add_args_testing_subj_hdf5

blue = [2., 75., 252.]


def prepare_args_visu_loss(p: ArgumentParser, use_existing_experiment=True):
    # Mandatory
    if use_existing_experiment:
        # Should only be False for debugging tests.
        add_arg_existing_experiment_path(p)
        p.add_argument('--uncompress_loss', action='store_true',
                       help="If model uses compressed loss, take uncompressed "
                            "equivalent.")
        p.add_argument('--force_compress_loss', nargs='?', type=float,
                       const=1e-3,
                       help="Compress loss, even if model uses uncompressed "
                            "loss.")
        g = p.add_mutually_exclusive_group()
        g.add_argument('--weight_with_angle', action='store_true',
                       help="Change model's weight loss with angle parameter "
                            "value (True/False).")
        g.add_argument('--do_not_weight_with_angle', dest='weight_with_angle',
                       action='store_false')

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
    print("Loss ranges between {} and {}".format(min_val, max_val))

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


def separate_best_and_worst(percent, losses, sft):
    losses = [np.mean(s_losses) for s_losses in losses]

    percent = int(percent / 100 * len(sft))
    idx = np.argsort(losses)
    best_idx = idx[0:percent]
    worst_idx = idx[-percent:]

    print("Best / worst streamline's loss: \n"
          "      Best : {}\n"
          "      Worst: {}".format(losses[best_idx[0]], losses[worst_idx[-1]]))
    return best_idx, worst_idx


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
        tmp = [[s[p] + streamline_out_dir[p], s[p+1]]
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


def run_visu_save_colored_displacement(
        args, model: ModelWithDirectionGetter, losses: List[torch.Tensor],
        outputs: List[torch.Tensor], sft: StatefulTractogram,
        colorbar_name: str, best_sft_name: str, worst_sft_name: str):

    if model.direction_getter.compress_loss:
        if not ('uncompress_loss' in args and args.uncompress_loss):
            print("Can't save colored SFT for compressed loss")

    # Save colored SFT
    if args.out_colored_sft is not None:
        logging.info("Preparing colored sft")
        sft, colorbar_fig = prepare_colors_from_loss(
            losses, sft, args.colormap, args.min_range, args.max_range)
        print("Saving colored SFT as {}".format(args.out_colored_sft))
        save_tractogram(sft, args.out_colored_sft)

        print("Saving colorbar as {}".format(colorbar_name))
        colorbar_fig.savefig(colorbar_name)

    # Separate best and worst
    best_idx = []
    worst_idx = []
    if args.save_best_and_worst is not None or args.pick_best_and_worst:
        best_idx, worst_idx = separate_best_and_worst(
            args.save_best_and_worst, losses, sft)

        if args.out_colored_sft is not None:
            best_sft = sft[best_idx]
            worst_sft = sft[worst_idx]
            print("Saving best and worst streamlines as {} \nand {}"
                  .format(best_sft_name, worst_sft_name))
            save_tractogram(best_sft, best_sft_name)
            save_tractogram(worst_sft, worst_sft_name)

    # Save displacement
    args.pick_idx = list(range(10))
    if args.out_displacement_sft:
        if args.out_colored_sft:
            # We have run model on all streamlines. Picking a few now.
            sft, idx = pick_a_few(
                sft, best_idx, worst_idx, args.pick_at_random,
                args.pick_best_and_worst, args.pick_idx)

            # ToDo. See if we can simplify to fit with all models
            if 'gaussian' in model.direction_getter.key:
                means, sigmas = outputs
                means = [means[i] for i in idx]
                lengths = [len(line) for line in means]
                outputs = (torch.vstack(means),
                           torch.vstack([sigmas[i] for i in idx]))

            elif 'fisher' in model.direction_getter.key:
                raise NotImplementedError
            else:
                outputs = [outputs[i] for i in idx]
                lengths = [len(line) for line in outputs]
                outputs = torch.vstack(outputs)

        # Use eos_thresh of 1 to be sure we don't output a NaN
        with torch.no_grad():
            out_dirs = model.get_tracking_directions(
                outputs, algo='det', eos_stopping_thresh=1.0)

            out_dirs = torch.split(out_dirs, lengths)

        out_dirs = [o.numpy() for o in out_dirs]

        # Save error together with ref
        sft = combine_displacement_with_ref(out_dirs, sft, model.step_size)

        save_tractogram(sft, args.out_displacement_sft, bbox_valid_check=False)

    if args.show_colorbar:
        plt.show()
