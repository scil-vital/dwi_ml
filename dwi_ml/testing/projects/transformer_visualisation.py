# -*- coding: utf-8 -*-
import logging
import os
import shutil
import subprocess

import torch

from scilpy.io.utils import assert_outputs_exist

from dwi_ml.models.projects.transforming_tractography import \
    OriginalTransformerModel, TransformerSrcAndTgtModel

# Currently, with our quite long sequences compared to their example, this
# is a bit ugly.
from dwi_ml.testing.projects.transformer_visualisation_utils import \
    load_data_run_model, tto_show_head_view, tto_show_model_view, \
    tto_prepare_tokens, ttst_prepare_tokens, ttst_show_model_view, \
    ttst_show_head_view

NORMALIZE_WEIGHTS = True
INFORMATIVE_TOKENS = False


def visualize_weights_using_jupyter(ipynb_filename, config_filename,
                                    parser, args, argv):
    # 1. Test that we correctly found the ipynb file.
    if not os.path.isfile(ipynb_filename):
        raise ValueError(
            "We could not find the jupyter notebook file. Probably a "
            "coding error on our side. We expected it to be in {}"
            .format(ipynb_filename))

    # 2. Verify that output dir exists but not output files.
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu')
    out_html_filename = args.out_prefix + 'tto_visu.html'
    out_html_file = os.path.join(args.out_dir, out_html_filename)
    out_ipynb_file = os.path.join(
        args.out_dir, args.out_prefix + 'tto_visu.ipynb')
    out_config_file = os.path.join(
        args.out_dir, args.out_prefix + 'tto_visu.config')

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)
    assert_outputs_exist(parser, args,
                         [out_html_file, out_ipynb_file, out_config_file])

    # 3. Save the args to this script in a config file, to be read again
    # by the notebook in a new argparser instance.
    if os.path.isfile(config_filename):  # In case a previous call failed.
        os.remove(config_filename)
    with open(config_filename, 'w') as f:
        f.write(' '.join(argv))

    # 4. Copy files locally
    shutil.copyfile(ipynb_filename, out_ipynb_file)
    shutil.copyfile(config_filename, out_config_file)

    # 5. Launching.
    print("\n\n"
          "********\n"
          "* We will run the jupyter notebook and save its result "
          "as a html file:\n"
          "* {}\n"
          "********\n\n".format(out_html_file))
    try:
        command = 'jupyter nbconvert --output-dir={:s} --output={} ' \
                  '--execute {:s} --to html' \
            .format(args.out_dir, out_html_filename, ipynb_filename)
        print("Running command:\n\n>>{}\n\n".format(command))
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError:
        print("JUPYTER NOTEBOOK EXCEUTION WAS NOT SUCCESSFULL.")
        print("You may try to use --run_locally to debug.")
        print("You may try to run yourself the notebook copied in "
              "--out_dir using the config file.")
        print("You may try to specify which kernel (i.e. which python "
              "version) \nyour jupyter notebooks should run on:\n"
              ">> (workon your favorite python environment)\n"
              ">> pip install ipykernel\n"
              ">> python -m ipykernel install --user\n")

    # 6. Delete config file.
    os.remove(config_filename)


def tto_visualize_weights(args, parser):
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    logging.info("Loading model.")
    model = OriginalTransformerModel.load_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)
    model.set_context('visu')

    weights, batch_streamlines = load_data_run_model(parser, args, model)
    sa_encoder, sa_decoder, mha_cross = weights

    # ----------------
    # Shape of attentions are a List of nb_layers x:
    #                [nb_streamlines, (nb_heads), batch_max_len, batch_max_len]
    # Reshaping all attentions to List of nb layers x
    #                [nb_streamlines=1, nb_heads, this_s_len,   this_s_len]
    # (nb heads dimension is skipped if args.average_heads).
    # -----------------
    nb_layers = len(sa_encoder)
    nb_heads = 1 if args.average_heads else model.nheads

    streamline1 = batch_streamlines[0]
    this_seq_len = len(streamline1)
    if model.direction_getter.add_eos:
        logging.info("Chosen streamline's length: {} points"
                     .format(len(streamline1)))
    else:
        logging.info("Chosen streamline's length (after removing the last "
                     "unused point): {} points".format(len(streamline1)))

    encoder_tokens, decoder_tokens = tto_prepare_tokens(streamline1)

    # Unpadding attention
    if args.average_heads:
        print("Averaging heads!")
        # Taking one streamline, one head = twice squeezed. Adding back the
        # two missing dimensions.
        encoder_attention = [
            layer[0, 0:this_seq_len, 0:this_seq_len][None, None, :, :]
            for layer in sa_encoder]
        decoder_attention = [
            layer[0, 0:this_seq_len, 0:this_seq_len][None, None, :, :]
            for layer in sa_decoder]
        cross_attention = [
            layer[0, 0:this_seq_len, 0:this_seq_len][None, None, :, :]
            for layer in mha_cross]

    else:
        print("Number of heads: Encoder: {}. Decoder: {}. Mixed: {}"
              .format(sa_encoder[0].shape[1], sa_decoder[0].shape[1],
                      mha_cross[0].shape[1]))
        # Taking one streamline = once squeezed. Adding back the missing
        # dimension.
        encoder_attention = [
            layer[0, :, 0:this_seq_len, 0:this_seq_len][None, :, :]
            for layer in sa_encoder]
        decoder_attention = [
            layer[0, :, 0:this_seq_len, 0:this_seq_len][None, :, :]
            for layer in sa_decoder]
        cross_attention = [
            layer[0, :, 0:this_seq_len, 0:this_seq_len][None, :, :]
            for layer in mha_cross]

    if NORMALIZE_WEIGHTS:
        print("Normalizing weights for each time point for a better "
              "rendering.")
        # Trying to perceive better: Weights normalized per point.
        encoder_attention = [torch.nn.functional.normalize(layer, dim=2)
                             for layer in encoder_attention]
        decoder_attention = [torch.nn.functional.normalize(layer, dim=2)
                             for layer in decoder_attention]
        cross_attention = [torch.nn.functional.normalize(layer, dim=2)
                           for layer in cross_attention]

    tto_show_head_view(encoder_attention, decoder_attention,
                       cross_attention, encoder_tokens, decoder_tokens)

    # Model view not useful if we have only one layer, one head.
    if nb_layers > 1 or nb_heads > 1:
        tto_show_model_view(encoder_attention, decoder_attention,
                            cross_attention, encoder_tokens, decoder_tokens,
                            nb_heads, nb_layers)


def ttst_visualize_weights(args, parser):
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    logging.info("Loading model.")
    model = TransformerSrcAndTgtModel.load_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)
    model.set_context('visu')

    weights, batch_streamlines = load_data_run_model(parser, args, model)
    sa_encoder, = weights

    # ----------------
    # Shape of attentions are a List of nb_layers x:
    #                [nb_streamlines, (nb_heads), batch_max_len, batch_max_len]
    # Reshaping all attentions to List of nb layers x
    #                [nb_streamlines=1, nb_heads, this_s_len,   this_s_len]
    # (nb heads dimension is skipped if args.average_heads).
    # -----------------
    nb_layers = len(sa_encoder)
    nb_heads = 1 if args.average_heads else model.nheads

    streamline1 = batch_streamlines[0]
    this_seq_len = len(streamline1)
    if model.direction_getter.add_eos:
        logging.info("Chosen streamline's length: {} points"
                     .format(len(streamline1)))
    else:
        logging.info("Chosen streamline's length (after removing the last "
                     "unused point): {} points".format(len(streamline1)))

    tokens = ttst_prepare_tokens(streamline1)

    # Unpadding attention
    if args.average_heads:
        # Taking one streamline, one head = twice squeezed. Adding back the
        # two missing dimensions.
        encoder_attention = [
            layer[0, 0:this_seq_len, 0:this_seq_len][None, None, :, :]
            for layer in sa_encoder]
    else:
        # Taking one streamline = once squeezed. Adding back the missing
        # dimension.
        encoder_attention = [
            layer[0, :, 0:this_seq_len, 0:this_seq_len][None, :, :]
            for layer in sa_encoder]

    if NORMALIZE_WEIGHTS:
        print("Normalizing weights for each time point for a better "
              "rendering.")
        # Trying to perceive better: Weights normalized per point.
        encoder_attention = [torch.nn.functional.normalize(layer, dim=2)
                             for layer in encoder_attention]

    ttst_show_head_view(encoder_attention, tokens)

    # Model view not useful if we have only one layer, one head.
    if nb_layers > 1 or nb_heads > 1:
        ttst_show_model_view(encoder_attention, tokens, nb_heads, nb_layers)
