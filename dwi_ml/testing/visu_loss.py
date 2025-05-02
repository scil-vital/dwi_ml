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
from dipy.io.streamline import save_tractogram, load_tractogram
from matplotlib import pyplot as plt

# toDo in scilpy2.0: use
# from scilpy.tractograms.dps_and_dpp_management import add_data_as_color_dpp

from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.testing.testers import load_sft_from_hdf5, \
    TesterWithDirectionGetter

blue = [2., 75., 252.]


def run_all_visu_loss(tester: TesterWithDirectionGetter,
                      model: ModelWithDirectionGetter, args, names):
    """
    Runs all steps of loss visualisation:
    1. Load SFT.
    2. Run model (through the tester, loops on all batches)
    3. Show histogram(s).
    4. Prepare and save colored SFT (loss per point)
    5. Prepare and save displacement SFT
    6. Prepare and save colored SFT (eos_probs per point)
    7. Prepare and save colored SFT (eos_errors per point)

    Parameters
    ----------
    tester: TesterWithDirectionGetter
        Instantiated tester
    model:ModelWithDirectionGetter
        Instantiated model
    args: Namespace
        With all our visu args
    names: Tuple[str]
        Results from visu_checks, to be called in the main of each script.
    """
    (histogram_name, histogram_name, histogram_name_eos_error,
     histogram_name_eos_probs_third,
     colored_sft_name, colorbar_name,
     colored_best_name, colored_worst_name,
     colored_eos_probs_name, colorbar_eos_probs_name,
     colored_eos_errors_name, colorbar_eos_errors_name,
     displacement_sft_name) = names

    # 1. Load SFT. Either from hdf5 or directly from file
    # Will be resampled by tester if needed.
    if args.streamlines_group:
        sft = load_sft_from_hdf5(args.subj_id, args.hdf5_file, args.subset,
                                 args.streamlines_group)
    else:
        # Header compatibility with hdf5 not checked.
        sft = load_tractogram(args.streamlines_file, args.reference)

    # (Subsample if possible; avoids running on all streamlines for no reason.)
    if not (args.save_colored_tractogram or args.save_colored_best_and_worst
            or args.save_colored_eos or args.compute_histogram):
        # Only remains save_displacement option.
        nb = args.save_displacement
        replace = True if nb > len(sft) else False
        rng = np.random.default_rng()
        idx = rng.choice(range(len(sft)), size=nb, replace=replace)
        logging.info("Selecting {} streamlines out of {} for visualisation "
                     "of the output direction.".format(len(idx), len(sft)))
        sft = sft[idx]

    # 2. Run model's forward method + compute_loss.
    logging.info("Running model on {} streamlines to compute loss."
                 .format(len(sft)))
    (sft, outputs, losses, mean_loss_per_line,
     eos_probs, eos_errors, mean_eos_error_per_line) = tester.run_model_on_sft(
        sft, compute_loss=True)

    if not model.direction_getter.add_eos:
        # We will not get a loss value nor an output for the last point of the
        # streamlines. Removing from sft.
        logging.warning("The model does not include a EOS value. Last point "
                        "of each streamline is discarded from output "
                        "tractograms.")
        sft.streamlines = [line[:-1] for line in sft.streamlines]

    # 3. Show histogram.
    if args.compute_histogram:
        logging.info("Preparing histogram")
        plot_histogram(losses, mean_loss_per_line,
                       eos_probs, eos_errors, mean_eos_error_per_line,
                       histogram_name, histogram_name_eos_error,
                       histogram_name_eos_probs_third,
                       args.fig_size)

    # 4. Prepare and save colored SFT (loss per point)
    if args.save_colored_tractogram or args.save_colored_best_and_worst:
        logging.info("Preparing colored tractogram, and/or best and worst "
                     "streamlines, colored with loss.")
        run_visu_save_colored_sft(
            losses, sft, 'losses',
            colorbar_name, args.colormap, args.min_range, args.max_range,
            save_whole_tractogram=args.save_colored_tractogram,
            colored_sft_name=colored_sft_name,
            save_separate_best_and_worst=args.save_colored_best_and_worst,
            best_sft_name=colored_best_name, worst_sft_name=colored_worst_name,
            mean_losses=mean_loss_per_line)
        # Remove dpp before continuing.
        del sft.data_per_point['losses']
        del sft.data_per_point['color']

    # 5. Prepare and save displacement SFT
    if args.save_displacement:
        run_visu_save_colored_displacement(
            model, outputs, sft, displacement_sft_name,
            nb=args.save_displacement)

    # 6. Prepare and save colored EOS probs
    #    (no option of best/worst saving)
    if args.save_colored_eos_probs:
        if model.direction_getter.add_eos:
            run_visu_save_colored_sft(
                eos_probs, sft, 'eos_probs',
                colorbar_eos_probs_name, 'viridis',
                args.min_range_eos_probs, args.max_range_eos_probs,
                save_whole_tractogram=args.save_colored_eos_probs,
                colored_sft_name=colored_eos_probs_name)
            # Remove dpp before continuing.
            del sft.data_per_point['eos_probs']
            del sft.data_per_point['color']
        else:
            logging.warning("No EOS used in this model. Ignoring option "
                            "--save_colored_eos_probs.")

    # 7. Prepare and save colored EOS errors
    #    (no option of best/worst saving)
    if args.save_colored_eos_errors:
        if model.direction_getter.add_eos:
            run_visu_save_colored_sft(
                eos_errors, sft, 'eos_errors',
                colorbar_eos_errors_name, 'viridis',
                args.min_range_eos_errors, args.max_range_eos_errors,
                save_whole_tractogram=args.save_colored_eos_errors,
                colored_sft_name=colored_eos_errors_name)
        else:
            logging.warning("No EOS used in this model. Ignoring option "
                            "--save_colored_eos_errors.")

    if args.show_now:
        plt.show()


def _prepare_colors_from_values(
        losses: List[np.ndarray], sft: StatefulTractogram, colormap: str,
        min_range: float = None, max_range: float = None):
    """
    Used by run_visu_save_colored_sft.
    """

    losses = np.concatenate(losses)

    # normalize between 0 and 1
    # todo in scilpy 2.0:
    #  sft, min_val, max_val = add_data_as_color_dpp()
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


def _pick_best_and_worst(nb, mean_losses):
    """
    Equivalent of argsort. But makes verifications.

    Parameters
    ----------
    nb: int
        Will take the top n and the bottom n.
    mean_losses: list
        The lost of each streamline.

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


def _create_colored_displacement(out_dirs, sft, model):
    """
    Used by run_visu_save_displacement.

    1. Normalizes directions, to get segments of length step size.
    2. Combines the displacement together with the input streamline:
        - This streamline is created by starting at the real position at each
        point, then advancing in the learned direction.
        - Between two points, we always add a point going back to the real
        position to view difference better. This means that the learned
        streamlines is twice the length of the real streamline, looking as a
        zigzag.
    3. Adds colors to the result.
    """
    # Prepare values
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

        # 1. Normalizing directions to step_size
        if _step_size_vox is not None:
            streamline_out_dir /= np.maximum(
                epsilon, np.linalg.norm(streamline_out_dir, axis=1)[:, None])
            streamline_out_dir *= _step_size_vox

        streamline_out_dir = [list(d) for d in streamline_out_dir]

        # 2. Preparing displacement.
        # output : Starts on first point
        #          + tmp = each point = true previous point + learned dir
        #                 + in between each point, comes back to correct point.
        tmp = [[s[p] + streamline_out_dir[p], s[p + 1]]
               for p in range(this_s_len - 1)]
        out_streamline = [s[0]] + list(itertools.chain.from_iterable(tmp))
        out_streamline = out_streamline[:-1]
        this_s_len2 = len(out_streamline)
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

    data_per_point = {
        'color_x': color_x,
        'color_y': color_y,
        'color_z': color_z
    }
    sft = sft.from_sft(out_streamlines, sft,
                       data_per_point=data_per_point)

    return sft


def _plot_whole_hist(fig_size, values, mean_per_line, value_name, nb_bins,
                     histogram_name):
    fig, axs = plt.subplots(3, 1)
    if fig_size is not None:
        fig.set_figheight(fig_size[0])
        fig.set_figwidth(fig_size[1])

    tmp = np.hstack(values)
    axs[0].hist(tmp, bins=nb_bins)
    axs[0].set_title("Histogram of the {} per point".format(value_name))
    axs[1].hist(tmp, bins=nb_bins, log=True)
    axs[1].set_title("Histogram of the {} per point (log norm)"
                     .format(value_name))
    axs[2].hist(mean_per_line, bins=nb_bins)
    axs[2].set_title("Histogram of the {} per streamline".format(value_name))
    logging.info("Saving histogram of the {} as {}"
                 .format(value_name, histogram_name))
    plt.savefig(histogram_name)


def _plot_log_hist_per_third(fig_size, values, value_name, nb_bins,
                             histogram_name):
    fig, axs = plt.subplots(4, 1)
    if fig_size is not None:
        fig.set_figheight(fig_size[0])
        fig.set_figwidth(fig_size[1])

    tmp1 = np.hstack([s[0:len(s) // 3] for s in values])
    axs[0].hist(tmp1, bins=nb_bins, log=True)
    axs[0].set_title("First third")
    axs[0].set_xlim([0, 1])

    tmp2 = np.hstack([s[len(s) // 3:2 * len(s) // 3] for s in values])
    axs[1].hist(tmp2, bins=nb_bins, log=True)
    axs[1].set_title("Second third")
    axs[1].set_xlim([0, 1])

    tmp3 = np.hstack([s[2 * len(s) // 3:-2] for s in values])
    axs[2].hist(tmp3, bins=nb_bins, log=True)
    axs[2].set_title("Last third")
    axs[2].set_xlim([0, 1])

    tmp4 = np.hstack([s[-1] for s in values])
    axs[3].hist(tmp4, bins=nb_bins, log=True)
    axs[3].set_title("Last point")
    axs[3].set_xlim([0, 1])

    plt.suptitle("Histogram of the {} per point in each third of the "
                 "streamlines (log norm).".format(value_name))
    logging.info("Saving histogram of the loss as {}".format(histogram_name))
    plt.savefig(histogram_name)


def plot_histogram(losses, mean_per_line,
                   eos_probs, eos_errors, mean_per_line_eos_errors,
                   histogram_name, histogram_name_eos_errors,
                   histogram_name_eos_probs_third,
                   fig_size=None):
    logging.info("Preparing histogram!")
    # bins=auto fails?? Out of memory. Setting bins.
    nb_bins_loss = 200
    nb_bins_eos = 40
    _plot_whole_hist(fig_size, losses, mean_per_line, 'loss', nb_bins_loss,
                     histogram_name)

    if mean_per_line_eos_errors is not None:
        _plot_whole_hist(fig_size, eos_errors, mean_per_line_eos_errors,
                         'EOS error', nb_bins_eos, histogram_name_eos_errors)

        _plot_log_hist_per_third(fig_size, eos_probs, 'EOS probs',
                                 nb_bins_eos, histogram_name_eos_probs_third)


def run_visu_save_colored_sft(
        values: List[np.ndarray], sft: StatefulTractogram, dpp_key: str,
        colorbar_name: str, colormap: str,
        min_range: float = None, max_range: float = None,
        # Whole tractogram options
        save_whole_tractogram=False, colored_sft_name: str = None,
        # Best, worst options
        save_separate_best_and_worst: float = None, best_sft_name=None,
        worst_sft_name=None, mean_losses=None):
    """
    Saves the losses as data per point.
    """
    assert save_whole_tractogram or save_separate_best_and_worst

    logging.info("Adding losses as data per point:")
    sft.data_per_point[dpp_key] = [s[:, None] for s in values]

    sft, colorbar_fig = _prepare_colors_from_values(
        values, sft, colormap, min_range, max_range)

    logging.info("Saving colorbar as {}".format(colorbar_name))
    colorbar_fig.savefig(colorbar_name)

    if save_whole_tractogram:
        logging.info("Saving colored data as {}\n"
                     "with dpp keys: {}"
                     .format(colored_sft_name,
                             list(sft.data_per_point.keys())))
        save_tractogram(sft, colored_sft_name)

    if save_separate_best_and_worst:
        nb = int(save_separate_best_and_worst / 100 * len(sft))
        best_idx, worst_idx = _pick_best_and_worst(nb, mean_losses)

        logging.info("Saving best and worst colored streamlines as {} \nand {}"
                     .format(best_sft_name, worst_sft_name))

        save_tractogram(sft[best_idx], best_sft_name)
        save_tractogram(sft[worst_idx], worst_sft_name)


def run_visu_save_colored_displacement(
        model: ModelWithDirectionGetter, outputs: List[torch.Tensor],
        sft: StatefulTractogram, displacement_sft_name: str, nb: int):
    """
    Saves the displacement, colored: from green (start) to pink (end).
    """
    # 1. Select a few streamlines (if not already done)
    if len(sft) > nb:
        replace = True if nb > len(sft) else False
        rng = np.random.default_rng()
        idx = rng.choice(range(len(sft)), size=nb, replace=replace)
        logging.info("Selecting {} streamlines out of {} for visualisation "
                     "of the output direction.".format(len(idx), len(sft)))
        sft = sft[idx]

        if 'gaussian' in model.direction_getter.key:
            # outputs = means, sigmas
            out0 = [outputs[0][i] for i in idx]
            out1 = [outputs[1][i] for i in idx]
            outputs = (torch.vstack(out0), torch.vstack(out1))
        else:
            outputs = [outputs[i] for i in idx]
            outputs = torch.vstack(outputs)

    # 2. Get out_dirs from model_outputs using the direction getter.
    # Use eos_thresh of 1 to be sure we don't output a NaN
    lengths = [len(s) for s in sft.streamlines]
    logging.info("Getting tracking directions from the model output.\n"
                 "Using EOS threshold 1 to avoid getting NANs.\n"
                 "We will get an output direction at each point even "
                 "tough the model would have rather stopped.")
    with torch.no_grad():
        out_dirs = model.get_tracking_directions(
            outputs, algo='det', eos_stopping_thresh=1.0)
        out_dirs = torch.split(out_dirs, lengths)

    out_dirs = [o.numpy() for o in out_dirs]

    # 3. Save error together with ref
    sft = _create_colored_displacement(out_dirs, sft, model)

    logging.info("Saving displacement as {}".format(displacement_sft_name))
    logging.info("Tractogram displacement: \n"
                 "  - In blue: The original streamline.\n"
                 "  - From dark green (start) to pink: displacement of "
                 "estimation at each time point. Color was added to help find "
                 "the direction of the streamline.")

    save_tractogram(sft, displacement_sft_name, bbox_valid_check=False)
