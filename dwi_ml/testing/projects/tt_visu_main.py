# -*- coding: utf-8 -*-

"""
Main part for the tt_visualize_weights, separated to be callable by the
jupyter notebook.
"""
import argparse
import glob
import logging
import os

import numpy as np
import torch
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
    OriginalTransformerModel, TransformerSrcAndTgtModel)
from dwi_ml.models.projects.transformers_utils import find_transformer_class
from dwi_ml.testing.projects.tt_visu_bertviz import (
    encoder_decoder_show_head_view, encoder_decoder_show_model_view,
    encoder_show_model_view, encoder_show_head_view)
from dwi_ml.testing.projects.tt_visu_colored_sft import (
    save_sft_with_attention_as_dpp)
from dwi_ml.testing.projects.tt_visu_matrix import show_model_view_as_imshow
from dwi_ml.testing.projects.tt_visu_utils import (
    prepare_encoder_tokens, prepare_decoder_tokens,
    reshape_unpad_rescale_attention)
from dwi_ml.testing.testers import TesterOneInput
from dwi_ml.testing.utils import add_args_testing_subj_hdf5


def set_out_dir_visu_weights_and_create_if_not_exists(args):
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu_weights')
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
        '--out_prefix', metavar='name',
        help="Prefix of the all output files. Do not include a path. "
             "Suffixes are: \n"
             "   1) 'as_matrices': tt_matrix_[encoder|decoder|cross].png.\n"
             "   2) 'bertviz': tt_bertviz.html, tt_bertviz.ipynb, "
             "tt_bertviz.config.\n"
             "   3) 'colored_sft': colored_sft.trk."
             "   4) 'bertviz_locally': None")
    p.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu_weights")

    p.add_argument(
        '--visu_type', required=True, nargs='+',
        choices=['as_matrices', 'bertviz', 'colored_sft', 'bertviz_locally'],
        help="Output option. Choose any number (at least one). \n"
             "   1) 'as_matrices': Show attention as matrices. \n"
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
    p.add_argument(
        '--rescale', action='store_true',
        help="If true, rescale to max 1 per row.")

    g = p.add_mutually_exclusive_group()
    g.add_argument('--uniformize_endpoints', action='store_true',
                   help="If set, align endpoints of the batch. Either this "
                        "or --inverse_uniformize_endpoints. \nProbably helps"
                        "visualisation with option --visu_type 'colored_sft'.")
    g.add_argument('--inverse_uniformize_endpoints', action='store_true',
                   help="If set, aligns endpoints and then reverses the "
                        "bundle.")
    g.add_argument('--flip', action='store_true',
                   help="If set, flip all streamlines.")
    p.add_argument('--axis', choices=['x', 'y', 'z'],
                   help='When uniformizing endpoints, match endpoints of the '
                        'streamlines along this axis. If not set: discover '
                        'the best axis (auto).'
                        '\nSUGGESTION: Commissural = x, Association = y, '
                        'Projection = z')

    p.add_argument('--resample_attention', type=int, metavar='nb',
                   help="Streamline will be sampled as decided by the model."
                        "However, \nattention will be resampled to fit better "
                        "in the html page. \n (Resampling is done by "
                        "averaging the attention every N points).\n"
                        "(only for bertviz and as_matrices")
    p.add_argument('--average_heads', action='store_true',
                   help="If true, resample all heads (per layer per "
                        "attention type).")

    p.add_argument('--batch_size', type=int, metavar='n',
                   help="Batch size in number of streamlines. If not set, "
                        "uses all streamlines in one batch.")
    add_memory_args(p)

    p.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrices on screen. Else, only "
                        "saves them.")
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
    args = set_out_dir_visu_weights_and_create_if_not_exists(args)
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
    model_dir = os.path.join(args.experiment_path, 'best_model')
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

    sft = sft[0:1]  # - ------------------------ DEBUGGING
    logging.debug("   Got {} streamlines.".format(len(sft)))

    if len(sft) > 1 and not save_colored_sft:
        # Taking only one streamline
        line_id = np.random.randint(0, len(sft), size=1)[0]
        logging.info("    Picking ONE streamlines at random to show with "
                     "bertviz / show as matrices: #{} / {}."
                     .format(line_id, len(sft)))
        sft = sft[[line_id]]

    if args.uniformize_endpoints or args.inverse_uniformize_endpoints:
        # Done in-place
        uniformize_bundle_sft(sft, args.axis,
                              swap=args.inverse_uniformize_endpoints)
    elif args.flip:
        sft.streamlines = [np.flip(line, axis=0) for line in sft.streamlines]

    logging.debug("Loading the data...")
    tester = TesterOneInput(
        model=model, hdf5_file=args.hdf5_file, subj_id=args.subj_id,
        subset_name=args.subset, volume_group=args.input_group,
        batch_size=args.batch_size, device=device)

    logging.debug("Running the model on the given bundle...")
    model.set_context('visu_weights')
    sft, outputs, _, _ = tester.run_model_on_sft(sft)

    # Resulting weights is a tuple of one list per attention type.
    # Each list is: one tensor per layer.
    _, weights = outputs

    if isinstance(model, OriginalTransformerModel):
        visu_fct = visu_encoder_decoder
    elif isinstance(model, TransformerSrcAndTgtModel):
        visu_fct = visu_encoder_only
    else:  # TransformerSrcOnlyModel
        visu_fct = visu_encoder_only

    visu_fct(weights, sft, model.direction_getter.add_eos,
             args.average_heads, args.resample_attention, args.rescale,
             save_colored_sft, run_bertviz, show_as_matrices, prefix_total)

    if args.show_now:
        plt.show()


def visu_encoder_decoder(
        weights, sft, has_eos: bool,
        average_heads: bool, resample_nb: int, rescale: bool,
        save_colored_sft: bool, run_bertviz: bool, show_as_matrices: bool,
        prefix_name: str):
    """
    Visualizing the 3 attentions.
    """
    encoder_attention, decoder_attention, cross_attention = weights

    if not has_eos:
        logging.warning("No EOS in model. Will ignore the last point per "
                        "streamline")
        sft.streamlines = [s[:-1, :] for s in sft.streamlines]
    lengths = [len(s) for s in sft.streamlines]

    encoder_attention = reshape_unpad_rescale_attention(
        encoder_attention, average_heads, lengths, rescale)
    decoder_attention = reshape_unpad_rescale_attention(
        decoder_attention, average_heads, lengths, rescale)
    cross_attention = reshape_unpad_rescale_attention(
        cross_attention, average_heads, lengths, rescale)

    if save_colored_sft:
        print("\n\n-------------- Preparing the data_per_point to color sft "
              "--------------")
        save_sft_with_attention_as_dpp(
            sft, lengths, prefix_name,
            (encoder_attention, decoder_attention, cross_attention),
            ('encoder', 'decoder', 'cross'))

    if run_bertviz or show_as_matrices:
        print("\n\n-------------- Preparing the attention as a matrix for one "
              "streamline --------------")
        if save_colored_sft:
            print("Choosing only one streamline from the bundle to show "
                  "as matrices/bertviz. Not ready yet for multiple "
                  "streamlines.")
            sft.streamlines = sft.streamlines[0]
        # Else we already chose one streamline before running the whole model.

        encoder_attention = encoder_attention[0]
        decoder_attention = decoder_attention[0]
        cross_attention = cross_attention[0]
        this_seq_len = lengths[0]

        encoder_attention, inds = resample_attention_one_line(
            encoder_attention, this_seq_len, resample_nb=resample_nb)
        decoder_attention, inds = resample_attention_one_line(
            decoder_attention, this_seq_len, resample_nb=resample_nb)
        cross_attention, inds = resample_attention_one_line(
            cross_attention, this_seq_len, resample_nb=resample_nb)

        encoder_tokens = prepare_encoder_tokens(this_seq_len, has_eos, inds)
        decoder_tokens = prepare_decoder_tokens(this_seq_len, inds)

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
                                      prefix_name + '_matrix_encoder',
                                      encoder_tokens, encoder_tokens)
            print("DECODER ATTENTION: ")
            show_model_view_as_imshow(decoder_attention,
                                      prefix_name + '_matrix_decoder',
                                      decoder_tokens, decoder_tokens)
            print("CROSS ATTENTION: ")
            show_model_view_as_imshow(cross_attention,
                                      prefix_name + '_matrix_cross_attention',
                                      encoder_tokens, decoder_tokens)


def visu_encoder_only(
        weights, sft, has_eos: bool,
        average_heads: bool, resample_nb: int, rescale: bool,
        save_colored_sft: bool, run_bertviz: bool, show_as_matrices: bool,
        prefix_name: str):
    """
    Visualizing one attention.

    colored_sft_name: str, or None if not save_colored_sft
    matrices_prefix: str, or None if not show_as_matrices
    """
    # Weights = a Tuple one 1 attention.
    # Encoder_attention: A list of one tensor per layer, of shape:
    #    [nb_streamlines, batch_max_len, batch_max_len] --> If averaged heads
    #    [nb_streamlines, nheads, batch_max_len, batch_max_len] --> Else.
    encoder_attention, = weights

    logging.info("   Preparing encoder attention")

    if not has_eos:
        logging.warning("No EOS in model. Will ignore the last point per "
                        "streamline")
        sft.streamlines = [s[:-1, :] for s in sft.streamlines]
    lengths = [len(s) for s in sft.streamlines]

    # Reshaping all
    encoder_attention = reshape_unpad_rescale_attention(
        encoder_attention, average_heads, lengths, rescale, resample_nb)
    if resample_nb:
        sft = resample_streamlines_num_points(sft, num_points=resample_nb)

    # Right now attention = list per streamline.
    #  (of a list per layer)
    if save_colored_sft:
        print("\n\n-------------- Preparing the data_per_point to color sft "
              "--------------")
        save_sft_with_attention_as_dpp(sft, lengths, prefix_name,
                                       (encoder_attention,), ('encoder',))

    if run_bertviz or show_as_matrices:
        print("\n\n-------------- Preparing the attention as a matrix for one "
              "streamline --------------")
        if save_colored_sft:
            print("Choosing only one streamline from the bundle to show "
                  "as matrices/bertviz. Not ready yet for multiple "
                  "streamlines.")
            sft.streamlines = sft.streamlines[0]
        # Else we already chose one streamline before running the whole model.

        encoder_attention = encoder_attention[0]
        this_seq_len = lengths[0]

        encoder_tokens = prepare_encoder_tokens(this_seq_len, has_eos)

        if show_as_matrices:
            print("ENCODER ATTENTION: ")
            show_model_view_as_imshow(encoder_attention,
                                      prefix_name + '_matrix_encoder',
                                      encoder_tokens)

        # Sending to 4D torch for Bertviz (each layer)
        encoder_attention = [torch.as_tensor(att)[None, :, :, :]
                             for att in encoder_attention]

        if run_bertviz:
            encoder_show_head_view(encoder_attention, encoder_tokens)
            encoder_show_model_view(encoder_attention, encoder_tokens)

