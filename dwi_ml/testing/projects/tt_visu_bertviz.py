# -*- coding: utf-8 -*-

# Ref: # https://github.com/jessevig/bertviz
from bertviz import model_view, head_view

# Currently, with our quite long sequences compared to their example, this
# is a bit ugly.
SHOW_MODEL_VIEW = False


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


def encoder_decoder_show_head_view(
        encoder_attention, decoder_attention, cross_attention,
        encoder_tokens, decoder_tokens):
    """
    Expecting attentions of shape:
        A list: nb_layers x
               [nb_streamlines, nheads, batch_max_len, batch_max_len]
    """
    print_head_view_help()
    head_view(encoder_attention=encoder_attention,
              decoder_attention=decoder_attention,
              cross_attention=cross_attention,
              encoder_tokens=encoder_tokens, decoder_tokens=decoder_tokens)


def encoder_show_head_view(encoder_attention, tokens):
    print_head_view_help()
    head_view(encoder_attention=encoder_attention, encoder_tokens=tokens)


def encoder_decoder_show_model_view(
        encoder_attention, decoder_attention, cross_attention,
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


def encoder_show_model_view(encoder_attention, tokens):
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
