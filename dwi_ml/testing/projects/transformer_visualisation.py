# -*- coding: utf-8 -*-
import argparse
import logging
import os

from scilpy.io.fetcher import get_home as get_scilpy_folder
from scilpy.io.utils import add_reference_arg, add_bbox_arg, add_overwrite_arg, assert_inputs_exist

from dwi_ml.io_utils import verify_which_model_in_path, add_logging_arg, \
    add_resample_or_compress_arg
from dwi_ml.models.projects.transformer_models import (
    OriginalTransformerModel, TransformerSrcOnlyModel,
    TransformerSrcAndTgtModel)

from dwi_ml.testing.projects.transformer_visualisation_submethods import (
    load_data_run_model, prepare_encoder_tokens, prepare_decoder_tokens,
    reformat_attention, tto_show_head_view, tto_show_model_view,
    ttst_show_model_view, ttst_show_head_view)

INFORMATIVE_TOKENS = False


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
    p.add_argument('--resample_attention', type=int, default=50,
                   help="Streamline will be sampled as decided by the model."
                        "However, \nattention will be resampled to fit better "
                        "in the html page. \n (Resampling is done by "
                        "averaging the attention every N points).\n"
                        "Default: 50")
    p.add_argument('--average_heads', action='store_true',
                   help="If true, resample all heads (per layer per "
                        "attention type).")
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

    add_resample_or_compress_arg(p)
    add_reference_arg(p)
    add_bbox_arg(p)
    add_logging_arg(p)
    add_overwrite_arg(p)

    return p


def tt_visualize_weights(args, parser):
    """
    Main part of the script: verifies with type of Transformer we have.
    """
    # Doing again the verification; in case this is run from the jupyter
    # notebook from an old config file.
    assert_inputs_exist(parser, [args.hdf5_file, args.input_streamlines],
                        args.reference)
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    sub_logger_level = 'WARNING'
    logging.getLogger().setLevel(level=args.logging)

    logging.info("1) Verifying model type.")
    model_dir = os.path.join(args.experiment_path, 'best_model')
    model_type = verify_which_model_in_path(model_dir)
    logging.info("   Model's class: {}\n".format(model_type))
    if model_type == 'OriginalTransformerModel':
        model_cls = OriginalTransformerModel
        visu_fct = tto_visualize_weights
    elif model_type == 'TransformerSrcAndTgtModel':
        model_cls = TransformerSrcAndTgtModel
        visu_fct = encoder_only_visualize_weights
    elif model_type == 'TransformerSrcOnlyModel':
        model_cls = TransformerSrcOnlyModel
        visu_fct = encoder_only_visualize_weights
    else:
        raise ValueError("Model type not a recognized transformer Transformer"
                         "({})".format(model_type))

    logging.info("2) Loading your model.\n")
    model = model_cls.load_model_from_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)
    model.set_context('visu')

    logging.info("3) Running the model on the bundle...")
    logging.info("   IMPORTANT !!! Right now the code is only ready to show "
                 "attention on one streamline at the time!")
    weights, batch_streamlines = load_data_run_model(
        parser, args, model, pick_one=True, sub_logger_level=sub_logger_level)
    logging.info("...Done!\n")

    logging.info("4) Preparing visu...")
    streamline1 = batch_streamlines[0]
    visu_fct(weights, streamline1, args.resample_attention,
             model.direction_getter.add_eos,)
    logging.info("... Done!")


def tto_visualize_weights(weights, streamline1, resample_attention: int,
                          add_eos: bool):

    # Original model: 3 attentions.
    encoder_attention, decoder_attention, cross_attention = weights

    # Unpadding
    this_seq_len = len(streamline1)

    logging.info("   Preparing encoder attention")
    encoder_attention, ind = reformat_attention(
        encoder_attention, this_seq_len, resample_attention)
    logging.info("   Preparing decoder attention")
    decoder_attention, _ = reformat_attention(
        decoder_attention, this_seq_len, resample_attention)
    logging.info("   Preparing cross-attention attention")
    cross_attention, _ = reformat_attention(
        cross_attention, this_seq_len, resample_attention)

    encoder_tokens = prepare_encoder_tokens(streamline1, add_eos, ind)
    decoder_tokens = prepare_decoder_tokens(streamline1, ind)

    tto_show_head_view(encoder_attention, decoder_attention,
                       cross_attention, encoder_tokens, decoder_tokens)
    tto_show_model_view(encoder_attention, decoder_attention,
                        cross_attention, encoder_tokens, decoder_tokens)


def encoder_only_visualize_weights(
        weights, streamline1, resample_attention: int, add_eos: bool):
    encoder_attention, = weights

    this_seq_len = len(streamline1)
    logging.info("   Preparing encoder attention (the only attention in this "
                 "model)")

    encoder_attention, ind = reformat_attention(
        encoder_attention, this_seq_len, resample_attention)

    encoder_tokens = prepare_encoder_tokens(streamline1, add_eos, ind)

    ttst_show_head_view(encoder_attention, encoder_tokens)
    ttst_show_model_view(encoder_attention, encoder_tokens)
