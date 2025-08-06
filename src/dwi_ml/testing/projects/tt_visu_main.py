# -*- coding: utf-8 -*-

"""
Main part for the tt_visualize_weights, separated to be callable by the jupyter
notebook.
"""
import glob
import logging
import os
from typing import Tuple

import numpy as np
import torch
from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.io_utils import verify_which_model_in_path
from dwi_ml.models.projects.transformer_models import find_transformer_class
from dwi_ml.testing.projects.tt_visu_bertviz import (
    encoder_decoder_show_head_view, encoder_decoder_show_model_view,
    encoder_show_model_view, encoder_show_head_view)
from dwi_ml.testing.projects.tt_visu_colored_sft import (
    color_sft_duplicate_lines, color_sft_x_y_projections)
from dwi_ml.testing.projects.tt_visu_matrix import show_model_view_as_imshow
from dwi_ml.testing.projects.tt_visu_utils import (
    prepare_encoder_tokens, prepare_decoder_tokens,
    reshape_unpad_rescale_attention, resample_attention_one_line,
    get_out_dir_and_create)
from dwi_ml.testing.testers import TesterOneInput


def tt_visualize_weights_main(args, parser):
    """
    Main part of the script: verifies with type of Transformer we have,
    loads the models, runs it to get the attention, and calls the right visu
    method.
    """
    # ------ Finalize parser verification
    if not (args.as_matrices or args.bertviz or args.colored_multi_length or
            args.colored_x_y_summary or args.bertviz_locally):
        parser.error("Expecting at least one visualisation option.")

    if args.resample_nb is not None and \
            not (args.as_matrices or args.bertviz or args.bertviz_locally):
        logging.warning("We only resample attention when visualizing matrices "
                        "or bertviz. Not required with current visualization "
                        "options. Ignoring.")

    average_heads = args.group_heads or args.group_all
    average_layers = args.group_all

    # -------- Verify inputs and outputs
    assert_inputs_exist(parser, [args.hdf5_file, args.in_sft])
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    # Out files: jupyter stuff already managed in main script. Remains the sft.
    # Whole filenames depend on rescaling options and grouping option.
    # Using the prefix to find any output.
    args = get_out_dir_and_create(args)
    prefix_total = os.path.join(args.out_dir, args.out_prefix)
    out_files = glob.glob(prefix_total + '*colored*.trk') + \
        glob.glob(prefix_total + '*.png')

    assert_outputs_exist(parser, args, out_files)
    if args.overwrite and len(out_files) > 0:
        # logging.warning("Removing these files from a previous run: {}"
        #                 .format(out_files))
        for f in out_files:
            if os.path.isfile(f):
                os.remove(f)

    sub_logger_level = 'WARNING'
    logging.getLogger().setLevel(level=args.verbose)

    if args.use_gpu:
        if torch.cuda.is_available():
            logging.debug("We will be using GPU!")
            device = torch.device('cuda')
        else:
            raise ValueError("You chose GPU (cuda) device but it is not "
                             "available!")
    else:
        device = torch.device('cpu')

    # ------------ Ok. Loading and formatting attention.
    sft, model, weights = _run_transformer_get_weights(
        parser, args, sub_logger_level, device)

    # ------------ Now show all
    _visu_encoder_decoder(
        weights, sft, model, average_heads, average_layers, args, prefix_total)

    if args.show_now:
        print("Showing matplotlib figures now. Everything is done, you can "
              "close figures manually or enter ctrl + c safely.")
        plt.show()


def _run_transformer_get_weights(parser, args, sub_logger_level, device):
    # 1. Load model
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

    # 2. Load SFT
    logging.info("Loading tractogram. Note that space comptability "
                 "with training data will NOT be verified.")
    args.bbox_check = False
    sft = load_tractogram_with_reference(parser, args, args.in_sft)
    sft.to_vox()
    sft.to_corner()
    logging.debug("   Got {} streamlines.".format(len(sft)))

    # 3. Preprocess SFT
    if len(sft) > 1 and not (
            args.color_multi_length or args.color_x_y_summary):
        # Taking only one streamline
        line_id = 0
        logging.info("    Picking THE FIRST streamline ONLY to show with "
                     "bertviz / show as matrices: #{} / {}."
                     .format(line_id, len(sft)))
        sft = sft[[line_id]]

    if args.reverse_lines:
        sft.streamlines = [np.flip(line, axis=0) for line in sft.streamlines]

    # 4. Load the rest of the data through the hdf5 (input_group and so on)
    logging.debug("Loading the input data from the hdf5...")
    tester = TesterOneInput(
        model=model, hdf5_file=args.hdf5_file, subj_id=args.subj_id,
        subset_name=args.subset, volume_group=args.input_group,
        batch_size=args.batch_size, device=device)

    # 5. Run the transformer.
    logging.debug("Running the model to get the weights...")
    model.set_context('visu_weights')
    sft, outputs, _, _, _, _, _ = tester.run_model_on_sft(
        sft, compute_loss=False)

    # Resulting weights is a tuple of one list per attention type.
    # Each list is: one tensor per layer.
    outputs, weights = outputs

    return sft, model, weights


def _visu_encoder_decoder(
        weights: Tuple, sft: StatefulTractogram, model,
        average_heads: bool, average_layers: bool, args, prefix_name: str):
    """
    Parameters
    ----------
    weights: Tuple[List]
        Either (encoder_attention,) or
               (encoder_attention, decoder_attention, cross_attention)
        Each attention is a list per layer, of tensors of shape
            [nb_streamlines, nheads, batch_max_len, batch_max_len]
    sft: StatefulTractogram
        The tractogram.
    model: AbstractTransformerModel
        The model.
    average_heads: bool
        Argparser's default = False
    average_layers: bool,
        Argparser's default = False. Must be False if average_head is False.
    args: Namespace
    prefix_name: str
        Includes the output path.
    """
    if len(weights) == 3:
        has_decoder = True
    else:
        has_decoder = False

    # 1. Prepare the streamlines
    has_eos = model.direction_getter.add_eos
    if not has_eos:
        logging.warning("No EOS in model. Will ignore the last point per "
                        "streamline")
        sft.streamlines = [s[:-1, :] for s in sft.streamlines]
    lengths = [len(s) for s in sft.streamlines]

    # 2. Arrange the weights
    weights = list(weights)
    explanation = None
    for i in range(len(weights)):
        weights[i], explanation = reshape_unpad_rescale_attention(
            weights[i], average_heads, average_layers, args.group_with_max,
            lengths, args.rescale_0_1, args.rescale_z, args.rescale_non_lin)

    if has_decoder:
        attention_names = ('encoder', 'decoder', 'cross')
    else:
        attention_names = ('encoder',)

    if args.color_multi_length:
        print(
            "\n-------------- Preparing the colors for each length of "
            "each streamline --------------")
        color_sft_duplicate_lines(sft, lengths, prefix_name, weights,
                                  attention_names, average_heads,
                                  average_layers, args.group_with_max,
                                  explanation)

    if args.color_x_y_summary:
        print(
            "\n-------------- Preparing the colors summary (nb_usage, "
            "where looked, etc) for each streamline --------------")
        color_sft_x_y_projections(
            sft, prefix_name, weights, attention_names,
            average_heads, average_layers, args.group_with_max,
            args.rescale_0_1, args.rescale_non_lin, args.rescale_z,
            explanation)

    if args.bertviz or args.as_matrices:
        if args.color_multi_length or args.color_x_y_summary:
            # Taking only one streamline. Was not done yet.
            line_id = 0
            logging.info("    Picking THE FIRST streamline ONLY to show with "
                         "bertviz / show as matrices: #{} / {}."
                         .format(line_id, len(sft)))
            sft = sft[[line_id]]
        # Else we already chose one streamline before running the whole model.

        name = prefix_name + '_single_streamline.trk'
        print("Saving the single line used for matrices, for debugging "
              "purposes, as ", name)
        save_tractogram(sft, name, bbox_valid_check=False)

        this_seq_len = lengths[0]
        for i in range(len(weights)):
            weights[i] = weights[i][0]
            weights[i] = resample_attention_one_line(
                weights[i], this_seq_len, args.resample_nb)

        step_size = np.linalg.norm(np.diff(sft.streamlines[0], axis=0), axis=1)
        step_size = np.mean(step_size)
        if args.resample_nb and this_seq_len > args.resample_nb:
            this_seq_len = args.resample_nb
        encoder_tokens = prepare_encoder_tokens(this_seq_len, step_size,
                                                has_eos)
        decoder_tokens = prepare_decoder_tokens(this_seq_len)
        weights_token = [(encoder_tokens, encoder_tokens),
                         (decoder_tokens, decoder_tokens),
                         (encoder_tokens, decoder_tokens)]

        if args.as_matrices:
            print(
                "\n\n-------------- Preparing the attention as a matrix for "
                "one streamline --------------")
            for i in range(len(weights)):
                print("Matrix for ", attention_names[i])
                show_model_view_as_imshow(
                    weights[i], prefix_name + '_matrix_' + attention_names[i],
                    *weights_token[i], args.rescale_0_1, args.rescale_z,
                    args.rescale_non_lin, average_heads, average_layers,
                    args.group_with_max)

        if args.bertviz or args.bertviz_locally:
            print(
                "\n\n-------------- Preparing the attention through bertviz "
                "for one streamline --------------")
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
