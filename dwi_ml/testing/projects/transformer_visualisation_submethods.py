# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import numpy as np
import torch

# Ref: # https://github.com/jessevig/bertviz
from bertviz import model_view, head_view
from matplotlib import pyplot as plt
from skimage.measure import block_reduce

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tractograms.streamline_operations import \
    resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft

from dwi_ml.models.projects.transformer_models import AbstractTransformerModel
from dwi_ml.testing.utils import prepare_dataset_one_subj

# Currently, with our quite long sequences compared to their example, this
# is a bit ugly.
SHOW_MODEL_VIEW = False


def load_data_run_model(parser, args, model: AbstractTransformerModel,
                        pick_one=True, sub_logger_level=None):
    """
    Pick_one: Decides what to do if SFT contains more than one streamline. If
    true: pick one. If false: to be decided.
    """

    # Prepare data
    logging.info("   Loading subject.")
    subset = prepare_dataset_one_subj(
        args.hdf5_file, args.subj_id, lazy=False, cache_size=1,
        subset_name=args.subset, volume_groups=[args.input_group],
        streamline_groups=[], log_level=sub_logger_level)

    # Load SFT
    logging.info("   Loading analysed bundle. Note that space comptability "
                 "with training data will NOT be verified.")
    sft = load_tractogram_with_reference(parser, args, args.input_streamlines)

    # Choosing 1 streamline
    if len(sft) > 1:
        if pick_one:
            streamline_ids = np.random.randint(0, len(sft), size=1)[0]
            logging.info("   Tractogram contained more than one streamline. "
                         "Picking any one: #{} / {}."
                         .format(streamline_ids, len(sft)))
        else:
            raise NotImplementedError
    else:
        streamline_ids = 0

    streamline_ids = 3544 #    ---------------------------------------------------------------- A EFFACER
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

    # OK TO DELETE?
    # if not model.direction_getter.add_eos:
    #    # We don't use the last coord because it does not have an
    #    # associated target direction.
    #    logging.info("Model used the last position of the streamline for EOS "
    #                 "learning. ")
    #    streamlines = [s[:-1, :] for s in streamlines]
    logging.info("   Loaded and prepared {} streamlines to be averaged for "
                 "visu.".format(len(streamlines)))
    if pick_one:
        if model.direction_getter.add_eos:
            logging.info("   Chosen streamline's length: {} points"
                         .format(len(streamlines[0])))
        else:
            logging.info("   Chosen streamline's length: {} + 1 points, but "
                         "the last one is not used because it does not have a "
                         "target direction.".format(len(streamlines[0])))

    # Prepare inputs
    group_idx = subset.volume_groups.index(args.input_group)
    subj_idx = 0
    batch_input = model.prepare_batch_one_input(
        streamlines, subset, subj_idx, group_idx)

    # Run model
    logging.info("   Loaded and prepared associated batch input. "
                 "Running model.")
    model.eval()
    grad_context = torch.no_grad()
    with grad_context:
        _, weights = model(batch_input, streamlines, return_weights=True,
                           average_heads=args.average_heads)
    return weights, streamlines


def reformat_attention(attention, this_seq_len, resample_attention: int):

    for ll in range(len(attention)):   # Per layer
        # Easier to work with numpy. Will put back to tensor after.
        att = attention[ll].numpy()

        # Unpadding. Taking one streamline. Adding back the dimension missing.
        if len(att.shape) == 3:
            # Shape of attentions are a List of nb_layers x:
            #   [nb_streamlines, batch_max_len, batch_max_len]
            att = att[0, 0:this_seq_len, 0:this_seq_len]
            att = att[None, None, :, :]
        else:
            # Shape of attentions are a List of nb_layers x:
            #   [nb_streamlines, nb_heads, batch_max_len, batch_max_len]
            att = att[0, :, 0:this_seq_len, 0:this_seq_len]
            att = att[None, :, :]

        # Right now shape of attention is:
        #     [nb_streamlines=1, nb_heads, this_s_len, this_s_len]
        # (nb_heads = 1 if average_heads).

        # Normalizing weight. Without it, we rapidly see nothing!
        # Easier to see when we normalize on the x axis.

        # Version 1) L_p pnormalization ==> V = V / norm(v)
        # attention[ll] = torch.nn.functional.normalize(attention[ll], dim=2)
        # attention[ll] = torch.nn.functional.normalize(attention[ll], dim=3)

        # Version 2) To percentages
        # Values seem already normalized in the last axis, so this is ~
        # equivalent to doing * 100. Bertviz seem to have a fixed colorscale
        # because this shows better.

        # Version 3)
        # Normalizing each row so that max value = 1.
        print("Normalizing attention at each point to range [0, 1]")
        max_ = np.max(att, axis=3)
        att = att / max_[:, :, :, None]


        if resample_attention < this_seq_len:
            print("RESAMPLING ATTENTION TO A SEQUENCE OF LENGTH {}\n"
                  "(uses the max per block)"
                  .format(resample_attention))

            nb_together = this_seq_len / resample_attention
            nb_together = int(np.floor(nb_together))

            ind1 = np.arange(0, this_seq_len, nb_together)
            ind = [(i, min(i + nb_together, this_seq_len - 1))
                   for i in ind1]

            # No option to pad to edge value in block_reduce. Doing manually.
            missing = (len(ind) * nb_together) - att.shape[2]
            att = np.pad(att, ((0, 0), (0, 0), (0, missing), (0, missing)),
                         mode='edge')

            att = block_reduce(
                att, block_size=(1, 1, nb_together, nb_together),
                func=np.max, cval=1000.0)  # 1000: to see if bug.

        else:
            ind = None

        attention[ll] = torch.as_tensor(att)

    return attention, ind


def prepare_decoder_tokens(streamline, ind: List[Tuple]):
    this_seq_len = len(streamline)

    if ind is not None:
        # Used resample_attention
        decoder_tokens = ['SOS-dir {}'.format(ind[0][1] - 2)] + \
                         ['dirs {} - {}'.format(i[0] - 1, i[0] - 2)
                          for i in ind]
    else:
        decoder_tokens = ['SOS'] + \
                         ['dir {}'.format(i) for i in range(this_seq_len - 1)]

    return decoder_tokens


def prepare_encoder_tokens(streamline, add_eos: bool, ind: List[Tuple]):
    # If encoder = concat X | Y, then , point0 = point0 | SOS
    #                                   point1 = point1 | dir0
    # etc. But ok. We will understand.
    this_seq_len = len(streamline)

    if ind is not None:
        # Used resample_attention
        encoder_tokens = ['points {}-{}'.format(i[0], i[1] - 1)
                          for i in ind]
    else:
        encoder_tokens = ['point {}'.format(i) for i in range(this_seq_len)]

    if add_eos:
        encoder_tokens[-1] += '(SOS)'

    return encoder_tokens


def print_head_view_help():
    print("\nINSTRUCTIONS of the head view: ")
    print(" -- Hover over any token on the left/right side of the "
          "visualization to filter attention from/to that token.")
    print(" -- Double-click on any of the colored tiles at the top to filter "
          "to the corresponding attention head.")
    print(" -- Single-click on any of the colored tiles to toggle selection "
          "of the corresponding attention head.")
    print(" -- Click on the Layer drop-down to change the model layer "
          "(zero-indexed).")


def print_model_view_help():
    print("\nINSTRUCTIONS of the model view for each head: ")
    print(" -- Click on any cell for a detailed view of attention for the "
          "associated attention head (or to unselect that cell).")
    print(" -- Then hover over any token on the left side of detail view to "
          "filter the attention from that token.")


def print_neuron_view_help():
    # Not used for now...
    print("\nINSTRUCTIONS of the neuron view:")
    print("Hover over any of the tokens on the left side of the visualization "
          "to filter attention from that token.")
    print("Then click on the plus icon that is revealed when hovering. This "
          "exposes the query vectors, key vectors, and other intermediate "
          "representations used to compute the attention weights. Each "
          "color band represents a single neuron value, where color "
          "intensity indicates the magnitude and hue the sign (blue=positive, "
          "orange=negative).")
    print("Once in the expanded view, hover over any other token on the left "
          "to see the associated attention computations.")
    print("Click on the Layer or Head drop-downs to change the model layer "
          "or head (zero-indexed).")


def tto_show_head_view(encoder_attention, decoder_attention, cross_attention,
                       encoder_tokens, decoder_tokens):
    print_head_view_help()
    head_view(encoder_attention=encoder_attention,
              decoder_attention=decoder_attention,
              cross_attention=cross_attention,
              encoder_tokens=encoder_tokens, decoder_tokens=decoder_tokens)


def show_model_view_as_imshow(attention, tokens_x, tokens_y=None):
    torch.set_printoptions(precision=2, sci_mode=False, linewidth=150)

    nb_layers = len(attention)
    size_x = len(tokens_x)

    if tokens_y is None:
        tokens_y = tokens_x.copy()
    size_y = len(tokens_y)

    cmap = plt.get_cmap("jet")
    cmap.set_bad(color='black')

    for i in range(nb_layers):
        att = attention[i].numpy()
        nb_heads = att.shape[1]
        fig, axs = plt.subplots(1, nb_heads, figsize=(20, 8))

        for h in range(nb_heads):
            a = np.squeeze(att[0, h, :, :])
            a = np.ma.masked_where(a == 0, a)
            im = axs[h].imshow(a, interpolation='None')
            axs[h].set_title("Head {}".format(h))
            axs[h].set_xticks(np.arange(size_x), fontsize=10)
            axs[h].set_yticks(np.arange(size_y), fontsize=10)
            axs[h].set_xticklabels(tokens_x, rotation=-90)
            axs[h].set_yticklabels(tokens_y)

            plt.colorbar(im, ax=axs[h])

        plt.suptitle("Layer {}\n"
                     "Each row: When getting next direction for this point, "
                     "what do we look at?\n"
                     "Each column: This data was used to decide which "
                     "directions?\n"
                     "DATA IS NORMALIZED TO [0-1] RANGE PER ROW".format(i))

        fig.tight_layout()

    plt.show()


def ttst_show_head_view(encoder_attention, tokens):
    print_head_view_help()
    head_view(encoder_attention=encoder_attention, encoder_tokens=tokens)


def tto_show_model_view(encoder_attention, decoder_attention, cross_attention,
                        encoder_tokens, decoder_tokens):
    # Supposing the same number of heads and layers for each attention
    nb_heads = encoder_attention[0].shape[1]
    nb_layers = len(encoder_attention)

    # Model view not useful if we have only one layer, one head.
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
            model_view(encoder_attention=tmp_e, decoder_attention=tmp_d,
                       cross_attention=tmp_c, encoder_tokens=encoder_tokens,
                       decoder_tokens=decoder_tokens)
    else:
        print("ENCODER ATTENTION: ")
        show_model_view_as_imshow(encoder_attention,
                                  encoder_tokens, encoder_tokens)
        print("DECODER ATTENTION: ")
        show_model_view_as_imshow(decoder_attention,
                                  decoder_tokens, decoder_tokens)
        print("CROSS ATTENTION: ")
        show_model_view_as_imshow(cross_attention,
                                  encoder_tokens, decoder_tokens)


def ttst_show_model_view(encoder_attention, tokens):
    # Supposing the same number of heads and layers for each attention
    nb_heads = encoder_attention[0].shape[1]
    nb_layers = len(encoder_attention)

    if SHOW_MODEL_VIEW:
        print_model_view_help()

        for head in range(nb_heads):
            print("HEAD #{}".format(head + 1))
            tmp_e = [encoder_attention[i][:, head, :, :][:, None, :, :]
                     for i in range(nb_layers)]
            model_view(encoder_attention=tmp_e, encoder_tokens=tokens)

    else:
        print("ENCODER ATTENTION: ")
        show_model_view_as_imshow(encoder_attention, tokens)