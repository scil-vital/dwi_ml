# -*- coding: utf-8 -*-
import argparse
import logging

import numpy as np
import torch

# Ref: # https://github.com/jessevig/bertviz
from bertviz import model_view, head_view

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, add_overwrite_arg, add_bbox_arg
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft

from dwi_ml.experiment_utils.prints import add_logging_arg
from dwi_ml.models.projects.transforming_tractography import \
    AbstractTransformerModel, OriginalTransformerModel
from dwi_ml.tracking.utils import prepare_dataset_one_subj

# Currently, with our quite long sequences compared to their example, this
# is a bit ugly.
SHOW_MODEL_VIEW = False
NORMALIZE_WEIGHTS = True
INFORMATIVE_TOKENS = False


def build_argparser_transformer_visu():
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
        '--out_dir',
        help="Output directory where to save the html file, as well as a \n"
             "copy of the jupyter notebook and config file.\n"
             "Default: experiment_path/visu")
    p.add_argument(
        '--out_prefix', default='',
        help="Prefix of the three output files. Names are tt*_visu.html, \n"
             "tt*_visu.ipynb and tt*_visu.config.")
    p.add_argument(
        '--run_locally', action='store_true',
        help="Run locally. Output will probably not show, but this is useful "
             "to debug.")

    add_reference_arg(p)
    add_bbox_arg(p)
    add_logging_arg(p)
    add_overwrite_arg(p)

    return p


def load_data_run_model(parser, args, model: AbstractTransformerModel,
                        pick_one=True):
    """
    Pick_one: Decides what to do if SFT contains more than one streamline. If
    true: pick one. If false: to be decided.
    """

    # Setting root logger to high level to max info, not debug, prints way too
    # much stuff. (but we can set our tracker's logger to debug)
    root_level = args.logging
    if root_level == logging.DEBUG:
        root_level = logging.INFO
    logging.getLogger().setLevel(level=root_level)

    # Prepare data
    args.lazy = False
    args.cache_size = None
    logging.info("Loading subject.")
    subset, subj_idx = prepare_dataset_one_subj(args)

    # Load SFT
    logging.info("Loading reference SFT.")
    sft = load_tractogram_with_reference(parser, args, args.input_streamlines)

    # Choosing 1 streamline
    if len(sft) > 1:
        if pick_one:
            streamline_ids = np.random.randint(0, len(sft), size=1)[0]
            logging.info("Tractogram contained more than one streamline. "
                         "Picking any one: #{}.".format(streamline_ids))
        else:
            raise NotImplementedError
    else:
        streamline_ids = 0

    # Bug in dipy. If accessing only one streamline, then sft.streamlines is
    # not a list of streamlines anymore, but the streamline itself. Thus,
    # len(sft) = nb_points instead of 1.
    streamline_ids = [streamline_ids]
    sft = sft[streamline_ids]

    # Resampling
    if args.step_size:
        # Resampling streamlines to a fixed step size, if any
        logging.debug("            Resampling: {}".format(args.step_size))
        sft = resample_streamlines_step_size(sft, step_size=args.step_size)
    if args.compress:
        logging.debug("            Compressing: {}".format(args.compress))
        sft = compress_sft(sft)

    # To tensor
    streamlines = [torch.as_tensor(s, dtype=torch.float32)
                   for s in sft.streamlines]
    streamlines = model.prepare_streamlines_f(streamlines)
    logging.info("Loaded and prepared {} streamlines to be averaged for visu."
                 .format(len(streamlines)))

    # Prepare inputs
    group_idx = subset.volume_groups.index(args.input_group)
    batch_input = model.prepare_batch_one_input(
        streamlines, subset, subj_idx, group_idx)

    # Run model
    logging.info("Loaded and prepared associated batch input. Running model.")
    model.eval()
    grad_context = torch.no_grad()
    with grad_context:
        _, weights = model(
            batch_input, streamlines,
            return_weights=True, average_heads=args.average_heads)
    return sft, weights


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

    sft, weights = load_data_run_model(parser, args, model)
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

    streamline1 = sft.streamlines[0]
    this_seq_len = len(streamline1) - 1
    logging.info("Chosen streamline's length: {} points"
                 .format(len(streamline1)))

    if INFORMATIVE_TOKENS:
        # Encoder: coordinates of the streamline, except last one
        encoder_tokens = ['p{} -- '.format(i) +
                          np.array2string(np.asarray(streamline1[i, :]),
                                          precision=1, separator=', ') +
                          ' -- p{}'.format(i)
                          for i in range(this_seq_len)]

        # Decoder input: directions, except the last one, plus SOS.
        dirs = np.diff(streamline1, n=1, axis=0)
        decoder_tokens = ['SOS'] + \
                         ['p{} -- '.format(i) +
                          np.array2string(np.asarray(dirs[i]),
                                          precision=2, separator=', ') +
                          ' -- p{} '.format(i)
                          for i in range(this_seq_len - 1)]
    else:
        # Just point1, point, etc.
        encoder_tokens = ['point {}'.format(i) for i in range(this_seq_len)]
        decoder_tokens = ['SOS'] + \
                         ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    if args.average_heads:
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


def print_head_view_help():
    print("INSTRUCTIONS of the head view: ")
    print(" -- Hover over any token on the left/right side of the visualization to filter attention from/to that token.")
    print(" -- Double-click on any of the colored tiles at the top to filter to the corresponding attention head.")
    print(" -- Single-click on any of the colored tiles to toggle selection of the corresponding attention head.")
    print(" -- Click on the Layer drop-down to change the model layer (zero-indexed).")


def print_model_view_help():
    print("INSTRUCTIONS of the model view for each head: ")
    print(" -- Click on any cell for a detailed view of attention for the associated attention head (or to unselect that cell).")
    print(" -- Then hover over any token on the left side of detail view to filter the attention from that token.")


def print_neuron_view_help():
    print("INSTRUCTIONS of the neuron view:")
    print("Hover over any of the tokens on the left side of the visualization to filter attention from that token.")
    print("Then click on the plus icon that is revealed when hovering. This exposes the query vectors, key vectors, "
          "and other intermediate representations used to compute the attention weights. Each color band represents "
          "a single neuron value, where color intensity indicates the magnitude and hue the sign (blue=positive, orange=negative).")
    print("Once in the expanded view, hover over any other token on the left to see the associated attention computations.")
    print("Click on the Layer or Head drop-downs to change the model layer or head (zero-indexed).")


def tto_show_head_view(encoder_attention, decoder_attention, cross_attention,
                       encoder_tokens, decoder_tokens):
    print_head_view_help()
    head_view(encoder_attention=encoder_attention,
              decoder_attention=decoder_attention,
              cross_attention=cross_attention,
              encoder_tokens=encoder_tokens,
              decoder_tokens=decoder_tokens)


def tto_show_model_view(encoder_attention, decoder_attention, cross_attention,
                        encoder_tokens, decoder_tokens, nb_heads, nb_layers):
    if SHOW_MODEL_VIEW:
        print_model_view_help()

        for head in range(nb_heads):
            print("HEAD #{}".format(head + 1))
            tmp_e = [encoder_attention[i][:, head, :, :][:, None, :, :]
                     for i in range(nb_layers)]
            tmp_d = [decoder_attention[i][:, head, :, :][:, None, :, :]
                     for i in range(nb_layers)]
            tmp_c = [cross_attention[i][:, head, :, :][:, None, :, :]
                     for i in range(nb_layers)]
            model_view(encoder_attention=tmp_e,
                       decoder_attention=tmp_d,
                       cross_attention=tmp_c,
                       encoder_tokens=encoder_tokens,
                       decoder_tokens=decoder_tokens)
