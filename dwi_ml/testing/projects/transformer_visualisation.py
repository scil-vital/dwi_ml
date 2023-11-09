# -*- coding: utf-8 -*-
import argparse
import logging
import os

import torch

from scilpy.io.fetcher import get_home as get_scilpy_folder
from scilpy.io.utils import add_reference_arg, add_bbox_arg, add_overwrite_arg, assert_inputs_exist

from dwi_ml.io_utils import verify_which_model_in_path, add_logging_arg
from dwi_ml.models.projects.transformer_models import \
    OriginalTransformerModel, TransformerSrcAndTgtModel

# Currently, with our quite long sequences compared to their example, this
# is a bit ugly.
from dwi_ml.testing.projects.transformer_visualisation_submethods import \
    load_data_run_model, tto_show_head_view, tto_show_model_view, \
    tto_prepare_tokens, ttst_prepare_tokens, ttst_show_model_view, \
    ttst_show_head_view

NORMALIZE_WEIGHTS = True
INFORMATIVE_TOKENS = False


def get_config_filename():
    # We choose to add it in the hidden .scilpy folder in our home.
    # (Where our test data also is).
    hidden_folder = get_scilpy_folder()
    config_filename = os.path.join(
        hidden_folder, 'ipynb_tt_visualize_weights.config')
    return config_filename


def build_argparser_transformer_visu():
    """
    This needs to be in a module, to be imported in the jupyter notebook. Do
    not put in the script file.
    """
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)

    p.add_argument('experiment_path',
                   help='Path to the directory containing the experiment.\n'
                        '(Should contain a model subdir with a file \n'
                        'parameters.json and a file best_model_state.pkl.)')
    p.add_argument('hdf5_file',
                   help="Path to the hdf5 file.")
    p.add_argument('subj_id',
                   help="Subject id to use for tractography.\n"
                        "Will also be added as prefix to add to the "
                        "out_tractogram name.")
    p.add_argument('input_group',
                   help="Input volume group name.")
    p.add_argument('input_streamlines',
                   help="A small tractogram; a bundle of streamlines whose "
                        "attention mask we will average.")
    p.add_argument('--subset', default='testing',
                   choices=['training', 'validation', 'testing'],
                   help="Subject id should probably come come the "
                        "'testing' set but you can\n modify this to "
                        "'training' or 'validation'.")
    sub = p.add_mutually_exclusive_group()
    sub.add_argument(
        '--step_size', type=float, metavar='s',
        help="Resample all streamlines to this step size (in mm). "
             "Default = None.")
    sub.add_argument(
        '--compress', action='store_true',
        help="If set, compress streamlines. Default = Not set.")
    p.add_argument('--average_heads', action='store_true',
                   help="If set, average heads when visualising outputs")

    p.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the html file, as well as a \n"
             "copy of the jupyter notebook and config file.\n"
             "Default: experiment_path/visu")
    p.add_argument(
        '--out_prefix', default='', metavar='name_',
        help="Prefix of the three output files. Names are tt*_visu.html, \n"
             "tt*_visu.ipynb and tt*_visu.config.")
    p.add_argument(
        '--run_locally', action='store_true',
        help="Run locally. Output will probably not show, but this is useful "
             "to debug. \nWithout this argument, we will run through jupyter "
             "to allow visualizing the results with Bert.")

    add_reference_arg(p)
    add_bbox_arg(p)
    add_logging_arg(p)
    add_overwrite_arg(p)

    return p


def tt_visualize_weights(args, parser):
    # Doing again the verification; in case this is run from the jupyter
    # notebook from an old config file.
    assert_inputs_exist(parser, [args.hdf5_file, args.input_streamlines],
                        args.reference)
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    logging.info("Verifying model type.")
    model_dir = os.path.join(args.experiment_path, 'best_model')
    model_type = verify_which_model_in_path(model_dir)
    print("Model's class: {}".format(model_type))
    if model_type == 'OriginalTransformerModel':
        visualize_weights = tto_visualize_weights
        jupyter = 'tto_visualize_weights.ipynb'
    elif model_type == 'TransformerSrcAndTgtModel':
        visualize_weights = ttst_visualize_weights
        jupyter = 'ttst_visualize_weights.ipynb'
    elif model_type == 'TransformerSrcOnlyModel':
        raise NotImplementedError("Not ready yet for this model.")
    else:
        raise ValueError("Model type not a recognized transformer Transformer"
                         "({})".format(model_type))


def tto_visualize_weights(args, parser):
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    logging.info("Loading model.")
    model = OriginalTransformerModel.load_model_from_params_and_state(
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
    model = TransformerSrcAndTgtModel.load_model_from_params_and_state(
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
