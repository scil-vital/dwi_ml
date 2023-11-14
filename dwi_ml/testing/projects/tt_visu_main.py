# -*- coding: utf-8 -*-

"""
Main part for the tt_visualize_weights, separated to be callable by the
jupyter notebook.
"""
import argparse
import logging
import os

import numpy as np
import torch
from dipy.io.streamline import save_tractogram

from scilpy.io.fetcher import get_home as get_scilpy_folder
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import (add_overwrite_arg, assert_inputs_exist,
                             assert_outputs_exist, add_reference_arg)

from dwi_ml.io_utils import add_logging_arg, verify_which_model_in_path, add_memory_args, \
    add_arg_existing_experiment_path
from dwi_ml.models.projects.transformer_models import (
    OriginalTransformerModel, TransformerSrcAndTgtModel)
from dwi_ml.models.projects.transformers_utils import find_transformer_class
from dwi_ml.testing.projects.tt_visu_bertviz import (
    encoder_decoder_show_head_view, encoder_decoder_show_model_view,
    encoder_show_model_view, encoder_show_head_view)
from dwi_ml.testing.testers import TesterOneInput
from dwi_ml.testing.projects.tt_visu_colored_sft import add_attention_as_dpp
from dwi_ml.testing.projects.tt_visu_matrix import show_model_view_as_imshow
from dwi_ml.testing.projects.tt_visu_utils import (
    prepare_encoder_tokens, prepare_decoder_tokens,
    reshape_attention, unpad_rescale_resample_attention)
from dwi_ml.testing.utils import add_args_testing_subj_hdf5


def set_out_dir_and_create_if_not_exists(args):
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu')
    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    return args


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

    add_arg_existing_experiment_path(p)
    add_args_testing_subj_hdf5(p, ask_input_group=True)

    p.add_argument('in_sft',
                   help="A small tractogram; a bundle of streamlines whose "
                        "attention mask we will average.")
    p.add_argument(
        '--visu_type', required=True, nargs='+',
        choices=['as_matrix', 'bertviz', 'colored_sft', 'bertviz_locally'],
        help="Output option. Choose any number (at least one). \n"
             "   1) 'as_matrix': Show attention as matrices. \n"
             "      If bertviz is also chosen, matrices will show in the "
             "html.\n"
             "   2) 'bertviz': Show using bertviz head_view visualization.\n"
             "      Will create a html file that can be viewed (see "
             "--out_dir)\n"
             "   3) 'colored_sft': Save a colored sft.\n"
             "   4) 'bertviz_locally': Run the bertviz without using jupyter "
             "(debug purposes).\n"
             "      Output will not not show, but html stuff will print in "
             "the terminal.")

    g = p.add_mutually_exclusive_group()
    g.add_argument('--align_endpoints', action='store_true',
                   help="If set, align endpoints of the batch. Either this "
                        "or --inverse_align_endpoints. Probably helps\n"
                        "visualisation with option --visu_type 'colored_sft'")
    g.add_argument('--inverse_align_endpoints', action='store_true',
                   help="If set, aligns endpoints and then reverses the "
                        "bundle.")
    p.add_argument('--resample_attention', type=int,
                   help="Streamline will be sampled as decided by the model."
                        "However, \nattention will be resampled to fit better "
                        "in the html page. \n (Resampling is done by "
                        "averaging the attention every N points).\n"
                        "(only for bertviz and as_matrix")
    p.add_argument('--average_heads', action='store_true',
                   help="If true, resample all heads (per layer per "
                        "attention type).")

    p.add_argument('--batch_size', type=int)
    add_memory_args(p)

    p.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu")
    p.add_argument(
        '--out_prefix', default='', metavar='name_',
        help="Prefix of the all output files. Files are: \n"
             "   1) 'as_matrix': None.\n"
             "   2) 'bertviz': tt_visu.html, tt_visu.ipynb, tt_visu.config.\n"
             "   3) 'colored_sft': colored_sft.trk."
             "   4) 'bertviz_locally': None")

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
    if 'as_matrix' in args.visu_type:
        show_as_matrices = True
    if 'bertviz' in args.visu_type or 'bertviz_locally' in args.visu_type:
        run_bertviz = True

    if save_colored_sft and (run_bertviz or show_as_matrices):
        raise NotImplementedError(
            "Currently, we run bertviz or show_as_matrices using one single "
            "streamline, but save colored_sft on all streamlines. Both "
            "options could be used but this is not ready.")
    if save_colored_sft and args.resample_attention is not None:
        parser.error("Do not use resample_attention with colored_sft.")

    # -------- Verify inputs and outputs
    assert_inputs_exist(parser, [args.hdf5_file, args.in_sft])
    if not os.path.isdir(args.experiment_path):
        parser.error("Experiment {} not found.".format(args.experiment_path))

    # Out files: jupyter stuff already managed in main script. Remains the sft.
    args = set_out_dir_and_create_if_not_exists(args)
    out_files = []
    out_sft = None
    if save_colored_sft:
        out_sft = os.path.join(args.out_dir,
                               args.out_prefix + 'colored_sft.trk')
        out_files.append(out_sft)

    assert_outputs_exist(parser, args, out_files)

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

    # Load SFT
    logging.info("Loading analysed bundle. Note that space comptability "
                 "with training data will NOT be verified.")
    sft = load_tractogram_with_reference(parser, args, args.in_sft)
    logging.debug("   Got {} streamlines.".format(len(sft)))

    # Load model
    logging.debug("Loading the model")
    model_dir = os.path.join(args.experiment_path, 'best_model')
    model_type = verify_which_model_in_path(model_dir)
    logging.debug("   Model's class: {}\n".format(model_type))
    model_cls = find_transformer_class(model_type)
    model = model_cls.load_model_from_params_and_state(
        model_dir, log_level=sub_logger_level)

    if len(sft) > 1 and not save_colored_sft:
        # Taking only one streamline
        line_id = np.random.randint(0, len(sft), size=1)[0]
        logging.info("    Picking ONE streamlines at random to show with "
                     "bertviz / show as matrices: #{} / {}."
                     .format(line_id, len(sft)))
        sft = sft[[line_id]]

    if args.align_endpoints:
        raise NotImplementedError

    if args.inverse_align_endpoints:
        raise NotImplementedError

    logging.debug("Loading the data...")
    tester = TesterOneInput(
        model=model, hdf5_file=args.hdf5_file, subj_id=args.subj_id,
        subset_name=args.subset, volume_group=args.input_group,
        batch_size=args.batch_size, device=device)

    logging.debug("Running the model on the given bundle...")
    model.set_context('visu_weights')
    sft, outputs, _, _ = tester.run_model_on_sft(sft)

    _, weights = outputs

    logging.info("Preparing visu...")
    if isinstance(model, OriginalTransformerModel):
        visu_fct = visu_encoder_decoder
    elif isinstance(model, TransformerSrcAndTgtModel):
        visu_fct = visu_encoder_only
    else:  # TransformerSrcOnlyModel
        visu_fct = visu_encoder_only
    visu_fct(weights, sft, args.resample_attention,
             model.direction_getter.add_eos, save_colored_sft,
             run_bertviz, show_as_matrices, colored_sft_name=out_sft)


def visu_encoder_decoder(
        weights, sft, resample_nb: int, add_eos: bool,
        save_colored_sft: bool, run_bertviz: bool, show_as_matrices: bool,
        colored_sft_name: str):
    """
    Visualizing the 3 attentions.
    """
    encoder_attention, decoder_attention, cross_attention = weights

    encoder_attention = reshape_attention(encoder_attention)
    decoder_attention = reshape_attention(decoder_attention)
    cross_attention = reshape_attention(cross_attention)

    if save_colored_sft:
        colored_sft = add_attention_as_dpp(sft, encoder_attention,
                                           'encoder_attention')
        colored_sft = add_attention_as_dpp(colored_sft, decoder_attention,
                                           'decoder_attention')
        colored_sft = add_attention_as_dpp(colored_sft, cross_attention,
                                           'cross_attention')
        save_tractogram(colored_sft, colored_sft_name)

    if run_bertviz or show_as_matrices:
        this_seq_len = len(sft[0])

        logging.info("   Preparing encoder attention")
        encoder_attention, ind = unpad_rescale_resample_attention(
            encoder_attention, [this_seq_len], resample_nb)
        logging.info("   Preparing decoder attention")
        decoder_attention, _ = unpad_rescale_resample_attention(
            decoder_attention, [this_seq_len], resample_nb)
        logging.info("   Preparing cross-attention attention")
        cross_attention, _ = unpad_rescale_resample_attention(
            cross_attention, [this_seq_len], resample_nb)

        encoder_tokens = prepare_encoder_tokens(this_seq_len, add_eos, ind)
        decoder_tokens = prepare_decoder_tokens(this_seq_len, ind)

        if run_bertviz:
            encoder_decoder_show_head_view(
                encoder_attention, decoder_attention, cross_attention,
                encoder_tokens, decoder_tokens)
            encoder_decoder_show_model_view(
                encoder_attention, decoder_attention, cross_attention,
                encoder_tokens, decoder_tokens)

        if show_as_matrices:
            print("ENCODER ATTENTION: ")
            show_model_view_as_imshow(encoder_attention,
                                      encoder_tokens, encoder_tokens)
            print("DECODER ATTENTION: ")
            show_model_view_as_imshow(decoder_attention,
                                      decoder_tokens, decoder_tokens)
            print("CROSS ATTENTION: ")
            show_model_view_as_imshow(cross_attention,
                                      encoder_tokens, decoder_tokens)


def visu_encoder_only(
        weights, sft, resample_nb: int, add_eos: bool,
        save_colored_sft: bool, run_bertviz: bool, show_as_matrices: bool,
        colored_sft_name: str):
    """
    Visualizing one attention.
    """
    # Weights = a Tuple one 1 attention.
    # Encoder_attention: A list of one tensor per layer, of shape:
    #    [nb_streamlines, batch_max_len, batch_max_len] --> If averaged heads
    #    [nb_streamlines, nheads, batch_max_len, batch_max_len] --> Else.
    encoder_attention, = weights

    logging.info("   Preparing encoder attention")
    encoder_attention = reshape_attention(encoder_attention)

    # Encoder_attention: A list of one tensor per layer
    #    [nb_streamlines, nheads, batch_max_len, batch_max_len]
    nb_layers = len(encoder_attention)
    nb_streamlines = len(sft)
    assert encoder_attention[0].shape[0] == nb_streamlines

    print("We will normalize the attention: per row, to the range [0, 1]")
    if save_colored_sft:
        # batch = all streamlines
        if not add_eos:
            logging.warning("No EOS in model. Will ignore the last point per "
                            "streamline")
            sft.streamlines = [s[:-1, :] for s in sft.streamlines]
        lengths = [len(s) for s in sft.streamlines]

        encoder_attention, inds = unpad_rescale_resample_attention(
            encoder_attention, lengths, resample_nb=None, for_bertviz=False)

        # Encoder_attention = A list of one list per layer:
        #   For each streamline:
        #       [nheads, this_seq_len, this_seq_len]
        assert len(encoder_attention) == nb_streamlines
        assert isinstance(encoder_attention[0], list)
        assert len(encoder_attention[0]) == nb_layers
        colored_sfts = add_attention_as_dpp(sft, encoder_attention, lengths,
                                            'encoder')

        save_tractogram(colored_sfts, colored_sft_name)

    elif run_bertviz or show_as_matrices:
        # batch = only one streamline
        this_seq_len = len(sft[0])

        encoder_attention, inds = unpad_rescale_resample_attention(
            encoder_attention, [this_seq_len], resample_nb)
        assert len(encoder_attention) == 1  # One streamline?
        encoder_attention = encoder_attention[0]

        ind = inds[0]
        encoder_tokens = prepare_encoder_tokens(this_seq_len, add_eos, ind)

        if run_bertviz:
            encoder_show_head_view(encoder_attention, encoder_tokens)
            encoder_show_model_view(encoder_attention, encoder_tokens)

        if show_as_matrices:
            print("ENCODER ATTENTION: ")
            show_model_view_as_imshow(encoder_attention, encoder_tokens)
