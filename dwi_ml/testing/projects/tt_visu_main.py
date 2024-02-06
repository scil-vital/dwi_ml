# -*- coding: utf-8 -*-

"""
Main part for the tt_visualize_weights, separated to be callable by the jupyter
notebook.
"""
import argparse
import glob
import logging
import os
from typing import Tuple

import numpy as np
import torch
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt

from scilpy.io.fetcher import get_home as get_scilpy_folder
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_reference_arg)
from scilpy.utils.streamlines import uniformize_bundle_sft

from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                             add_logging_arg, add_memory_args,
                             verify_which_model_in_path)
from dwi_ml.models.projects.transformer_models import (
    find_transformer_class, OriginalTransformerModel)
from dwi_ml.testing.projects.tt_visu_bertviz import (
    encoder_decoder_show_head_view, encoder_decoder_show_model_view,
    encoder_show_model_view, encoder_show_head_view)
from dwi_ml.testing.projects.tt_visu_colored_sft import (
    save_sft_with_attention_as_dpp)
from dwi_ml.testing.projects.tt_visu_matrix import show_model_view_as_imshow
from dwi_ml.testing.projects.tt_visu_utils import (
    prepare_encoder_tokens, prepare_decoder_tokens,
    reshape_unpad_rescale_attention, resample_attention_one_line)
from dwi_ml.testing.testers import TesterOneInput
from dwi_ml.testing.utils import add_args_testing_subj_hdf5


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


def create_out_dir_visu_weights(args):
    # Define out_dir as experiment_path/visu_weights if not defined.
    # Create it if it does not exist.
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu_weights')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    return args


def build_argparser_transformer_visu():
    """
    This needs to be in a module, to be imported in the jupyter notebook. Do
    not put in the script file.
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    # -------------
    # Positional args: Args to load data
    # -------------
    add_arg_existing_experiment_path(p)
    add_args_testing_subj_hdf5(p, ask_input_group=True)
    p.add_argument('in_sft',
                   help="A small tractogram; a bundle of streamlines that "
                        "should be \nuniformized. Else, see option "
                        "--align_endpoints")

    # --------------
    # Args to load data
    # --------------
    g = p.add_argument_group("Loading options")
    g.add_argument('--uniformize_endpoints', action='store_true',
                   help="If set, try aligning endpoints of the sft. Will use "
                        "the automatic \nalignment. For more options, align "
                        "your streamlines first, using \n"
                        "  >> scil_tractogram_uniformize_endpoints.py.\n")
    g.add_argument('--reverse_lines', action='store_true',
                   help="If set, reverses all streamlines first.\n"
                        "(With option --uniformize_endpoints, reversing is "
                        "done after.)")

    # --------------
    # Args to save data
    # --------------
    g = p.add_argument_group("Saving options")
    g.add_argument(
        '--out_prefix', metavar='name',
        help="Prefix of the all output files. Do not include a path. "
             "Suffixes are: \n"
             "   1) 'as_matrices': tt_matrix_[encoder|decoder|cross].png.\n"
             "   2) 'bertviz': tt_bertviz.html, tt_bertviz.ipynb, "
             "tt_bertviz.config.\n"
             "   3) 'colored_sft': colored_sft.trk.\n"
             "   4) 'bertviz_locally': None")
    g.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu_weights")

    # --------------
    # Options
    # --------------
    g = p.add_argument_group(
        "Weights processing options",
        description="By default, the sum per row in the matrix is 1. "
                    "Meaning: If all positions are equally important, their "
                    "value will be 1/n in the nth row.")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--rescale_0_1', action='store_true',
                    help="If set, rescale to max 1 per row. X = X/max(row)")
    gg.add_argument('--rescale_z', action='store_true',
                    help="If set, transform each value to X = (X - mu) / std, "
                         "where \nmu and std are computed per row.")
    gg.add_argument('--rescale_non_lin', action='store_true',
                    help="If set, transform each value so that values below "
                         "the equal \nattention are transformed to [0-0.5], "
                         "and values above to [0.5, 1].\n"
                         "(Ex: At point #3, 0.33 --> 0.5. At point #40, "
                         "0.025 --> 0.5.")

    g = p.add_argument_group("Options defining how to deal with the heads")
    gg = g.add_mutually_exclusive_group()
    gg.add_argument('--group_heads', action='store_true',
                    help="If true, average all heads (per layer, per "
                         "attention type).\n"
                         "To regroup using maximum instead, use "
                         "--group_with_max")
    gg.add_argument('--group_all', action='store_true',
                    help="If true, average all heads in all layers (per "
                         "attention type).\n"
                         "To regroup using maximum instead, use "
                         "--group_with_max")
    g.add_argument('--group_with_max', action='store_true',
                   help="Default grouping option is to average heads. Use "
                        "this option to group \nhead using their maximal "
                        "values.\n"
                        "NOTE: Average is done BEFORE rescaling (averaging "
                        "the raw weight),\nand max is done AFTER rescaling "
                        "(max of the rank use usefullness).")

    g = p.add_argument_group("Visualization options")
    g.add_argument(
        '--visu_type', required=True, nargs='+', metavar='visu',
        choices=['as_matrices', 'bertviz', 'colored_sft', 'bertviz_locally'],
        help="Output option. Choose any number (at least one). \n"
             "Currently, options bertviz and as_matrices use only the first \n"
             "streamline in the data.\n"
             "   1) 'as_matrices': Show attention as matrices. \n"
             "      If bertviz is also chosen, matrices will show in the "
             "html.\n"
             "   2) 'bertviz': Show using bertviz head_view visualization.\n"
             "      Will create a html file that can be viewed (see "
             "--out_dir)\n"
             "   3) 'colored_sft': Save a colored sft. Streamlines are "
             "duplicated\nat all lengths. Color of the streamline of length n "
             "is the weight \nof each point when getting the next direction "
             "at point n.\n"
             "   4) 'colored_mean': Save a colored sft. Color is the "
             "equivalent of the mean of the matrix on the vertical axis.\n"
             "   5) 'bertviz_locally': Run the bertviz without using jupyter\n"
             "      (Debugging purposes. Output will not not show, but html\n"
             "       stuff will print in the terminal.")
    g.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrices on screen. Else, only "
                        "saves them.")
    g.add_argument('--resample_plots', type=int, metavar='nb',
                   dest='resample_attention',
                   help="Streamlines will be sampled (nb points) as decided "
                        "by the model. \nHowever, attention shown as a matrix "
                        "can be resampled \nto better fit in the html page.")

    g = add_memory_args(p)
    g.add_argument('--batch_size', type=int, metavar='n',
                   help="Batch size in number of streamlines. If not set, "
                        "uses all streamlines \nin one batch.")
    add_reference_arg(p)
    add_logging_arg(p)
    add_overwrite_arg(p)

    return p


def tt_visualize_weights_main(args, parser):
    """
    Main part of the script: verifies with type of Transformer we have,
    loads the models, runs it to get the attention, and calls the right visu
    method.
    """
    # ------ Finalize parser verification
    save_colored_sft = False
    run_bertviz = False
    show_as_matrices = False
    if 'colored_sft' in args.visu_type:
        save_colored_sft = True
    if 'as_matrices' in args.visu_type:
        show_as_matrices = True
    if 'bertviz' in args.visu_type or 'bertviz_locally' in args.visu_type:
        run_bertviz = True

    if save_colored_sft and not (show_as_matrices or run_bertviz) and \
            args.resample_attention is not None:
        logging.warning("We only resample attention when visualizing matrices "
                        "or bertviz. Not required with colored_sft. Ignoring.")

    # -------- Verify inputs and outputs
    assert_inputs_exist(parser, [args.hdf5_file, args.in_sft])
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    # Out files: jupyter stuff already managed in main script. Remains the sft.
    args = create_out_dir_visu_weights(args)
    out_files = []
    prefix_total = os.path.join(args.out_dir, args.out_prefix)
    if save_colored_sft:
        # Total sft names will be, ex:
        # prefix_total + _colored_sft_encoder_layerX_headX.trk
        any_existing = glob.glob(prefix_total + '_colored_sft_*.trk')
        out_files.extend(any_existing)
    if show_as_matrices:
        # Total matrices names will be, ex:
        # prefix_total + _matrix_encoder_layerX_headX.png
        any_existing = glob.glob(prefix_total + '_matrix_*.png')
        out_files.extend(any_existing)

    assert_outputs_exist(parser, args, out_files)
    if args.overwrite:
        for f in out_files:
            if os.path.isfile(f):
                os.remove(f)

    sub_logger_level = 'WARNING'
    logging.getLogger().setLevel(level=args.logging)

    # ------------ Ok. Loading and formatting attention.
    if args.use_gpu:
        if torch.cuda.is_available():
            logging.debug("We will be using GPU!")
            device = torch.device('cuda')
        else:
            raise ValueError("You chose GPU (cuda) device but it is not "
                             "available!")
    else:
        device = torch.device('cpu')

    # Load model
    logging.debug("Loading the model")
    if args.use_latest_epoch:
        model_dir = os.path.join(args.experiment_path, 'best_model')
    else:
        model_dir = os.path.join(args.experiment_path, 'checkpoint/model')

    model_type = verify_which_model_in_path(model_dir)
    logging.debug("   Model's class: {}\n".format(model_type))
    model_cls = find_transformer_class(model_type)
    model = model_cls.load_model_from_params_and_state(
        model_dir, log_level=sub_logger_level)

    # Load SFT
    logging.info("Loading analysed bundle. Note that space comptability "
                 "with training data will NOT be verified.")
    sft = load_tractogram_with_reference(parser, args, args.in_sft)
    sft.to_vox()
    sft.to_corner()
    logging.debug("   Got {} streamlines.".format(len(sft)))

    if len(sft) > 1 and not save_colored_sft:
        # Taking only one streamline
        line_id = np.random.randint(0, len(sft), size=1)[0]
        logging.info("    Picking ONE streamlines at random to show with "
                     "bertviz / show as matrices: #{} / {}."
                     .format(line_id, len(sft)))
        sft = sft[[line_id]]

    if args.uniformize_endpoints:
        # Done in-place
        uniformize_bundle_sft(sft)

    if args.reverse_lines:
        sft.streamlines = [np.flip(line, axis=0) for line in sft.streamlines]

    logging.debug("Loading the data...")
    tester = TesterOneInput(
        model=model, hdf5_file=args.hdf5_file, subj_id=args.subj_id,
        subset_name=args.subset, volume_group=args.input_group,
        batch_size=args.batch_size, device=device)

    logging.debug("Running the model on the given bundle...")
    model.set_context('visu_weights')
    sft, outputs, _, _ = tester.run_model_on_sft(sft, compute_loss=False)

    # Resulting weights is a tuple of one list per attention type.
    # Each list is: one tensor per layer.
    outputs, weights = outputs

    if isinstance(model, OriginalTransformerModel):
        has_decoder = True
    else:
        has_decoder = False

    average_heads = args.group_heads or args.group_all
    average_layers = args.group_all
    if args.group_with_max and not \
            (args.rescale_0_1 or args.rescale_z or args.rescale_non_lin):
        parser.error("--group_with_max is expected to be used together with "
                     "a rescaling option.")
    visu_encoder_decoder(
        weights, sft, model.direction_getter.add_eos, average_heads,
        average_layers, args.group_with_max, args.resample_attention,
        args.rescale_0_1, args.rescale_z, args.rescale_non_lin,
        save_colored_sft, run_bertviz, show_as_matrices, prefix_total,
        has_decoder=has_decoder)

    if args.show_now:
        plt.show()


def visu_encoder_decoder(
        weights: Tuple, sft: StatefulTractogram, has_eos: bool,
        average_heads: bool, average_layers: bool, group_with_max: bool,
        resample_nb: int, rescale_0_1: bool, rescale_z: bool,
        rescale_non_lin: bool,
        save_colored_sft: bool, run_bertviz: bool, show_as_matrices: bool,
        prefix_name: str, has_decoder: bool = True):
    """
    Visualizing the 3 attentions.

    Parameters
    ----------
    weights: Tuple
        Either (encoder_attention,) or
               (encoder_attention, decoder_attention, cross_attention)
        Each attention is a list per layer, of tensors of shape
            [nb_streamlines, nheads, batch_max_len, batch_max_len]
    sft: StatefulTractogram
        The tractogram.
    has_eos: bool
        Whether the model uses EOS. Else, we have no weights for the last
        position.
    average_heads: bool
        Argparser's default = False
    average_layers: bool,
        Argparser's default = False. Must be False if average_head is False.
    group_with_max: bool
        Argparser's default = False (i.e. use average)
    resample_nb: int
        Number of values to resample matrices
    rescale_0_1: bool,
        If true, rescale each line of the matrix using X / max(X).
    rescale_z: bool
        If true, rescale each line of the matrix using (X-mu)/std
    rescale_non_lin: bool
        If true, rescale each line of the matrix to [0 - 0.5] and [0.5 - 1]
    save_colored_sft: bool
    run_bertviz: bool
        For now, on one streamline.
    show_as_matrices: bool
        For now, on one streamline.
    prefix_name: str
    has_decoder: bool
    """
    if average_layers:
        assert average_heads

    if not has_eos:
        logging.warning("No EOS in model. Will ignore the last point per "
                        "streamline")
        sft.streamlines = [s[:-1, :] for s in sft.streamlines]
    lengths = [len(s) for s in sft.streamlines]

    weights = list(weights)
    for i in range(len(weights)):
        weights[i] = reshape_unpad_rescale_attention(
            weights[i], average_heads, average_layers, group_with_max,
            lengths, rescale_0_1, rescale_z, rescale_non_lin)

    if has_decoder:
        attention_names = ('encoder',)
    else:
        attention_names = ('encoder', 'decoder', 'cross')

    if save_colored_sft:
        print("\n\n-------------- Preparing the data_per_point to color sft "
              "--------------")
        save_sft_with_attention_as_dpp(
            sft, lengths, prefix_name, weights, attention_names,
            average_heads, average_layers, group_with_max)

    if run_bertviz or show_as_matrices:
        if save_colored_sft:
            print("Choosing only one streamline from the bundle to show "
                  "as matrices/bertviz. Not ready yet for multiple "
                  "streamlines.")
            sft.streamlines = [sft.streamlines[0]]
        # Else we already chose one streamline before running the whole model.

        name = prefix_name + '_single_streamline.trk'
        print("Saving the single line used for matrices, for debugging "
              "purposes, as ", name)
        save_tractogram(sft, name)

        print("\n\n-------------- Preparing the attention as a matrix for one "
              "streamline --------------")
        this_seq_len = lengths[0]
        for i in range(len(weights)):
            weights[i] = weights[i][0]
            weights[i] = resample_attention_one_line(
                weights[i], this_seq_len, resample_nb)

        step_size = np.linalg.norm(np.diff(sft.streamlines[0], axis=0), axis=1)
        step_size = np.mean(step_size)
        if resample_nb and this_seq_len > resample_nb:
            this_seq_len = resample_nb
        encoder_tokens = prepare_encoder_tokens(this_seq_len, step_size,
                                                has_eos)
        decoder_tokens = prepare_decoder_tokens(this_seq_len)
        weights_token = [(encoder_tokens, encoder_tokens),
                         (decoder_tokens, decoder_tokens),
                         (encoder_tokens, decoder_tokens)]

        if show_as_matrices:
            for i in range(len(weights)):
                print("Matrix for ", attention_names[i])
                show_model_view_as_imshow(
                    weights[i], prefix_name + '_matrix_' + attention_names[i],
                    *weights_token[i], rescale_0_1,
                    average_heads, average_layers, group_with_max)

        if run_bertviz:
            # Sending to 4D torch for Bertviz (each layer)
            for i in range(len(weights)):
                weights[i] = [torch.as_tensor(att)[None, :, :, :]
                              for att in weights[i]]

            if has_decoder:
                encoder_decoder_show_head_view(
                    *weights, encoder_tokens, decoder_tokens)
                encoder_decoder_show_model_view(
                    *weights, encoder_tokens, decoder_tokens)
            else:
                encoder_show_head_view(*weights, encoder_tokens)
                encoder_show_model_view(*weights, encoder_tokens)
