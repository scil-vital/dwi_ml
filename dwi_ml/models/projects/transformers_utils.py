# -*- coding: utf-8 -*-
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.projects.positional_encoding import (
    keys_to_positional_encodings)
from dwi_ml.models.projects.transforming_tractography import (
    AbstractTransformerModel)
from dwi_ml.models.utils.direction_getters import check_args_direction_getter

sphere_choices = ['symmetric362', 'symmetric642', 'symmetric724',
                  'repulsion724', 'repulsion100', 'repulsion200']


def add_abstract_model_args(p):
    """ Optional parameters for TransformingTractography"""
    # Step_size / compress
    AbstractTransformerModel.add_args_main_model(p)

    gx = p.add_argument_group("Embedding:")
    gx.add_argument(
        '--data_embedding', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")
    gx.add_argument(
        '--position_encoding', default='sinusoidal', metavar='key',
        choices=keys_to_positional_encodings.keys(),
        help="Type of positional embedding to use. One of 'sinusoidal' "
             "(default)\n or 'relational'. ")
    gx.add_argument(
        '--token_type', default='as_label',
        choices=['as_label'] + sphere_choices,
        help="Type of token. SOS is always added in the decoder. EOS not used "
             "as token in the decoder. Choices are as_label (a "
             "fourth dimension) or \nas class (directions are sent to classes "
             "on the chosen sphere, and \nan additional class is added for "
             "SOS.")
    gx.add_argument(
        '--target_embedding', default='nn_embedding',
        choices=keys_to_embeddings.keys(), metavar='key',
        help="Type of data embedding to use. One of 'no_embedding', \n"
             "'nn_embedding' (default) or 'cnn_embedding'.")

    gt = p.add_argument_group(title='Transformer')
    gt.add_argument(
        '--d_model', type=int, default=4096, metavar='n',
        help="Output size that will kept constant in all layers to allow \n"
             "skip connections (embedding size, ffnn output size, attention \n"
             "size). [%(default)s]")
    gt.add_argument(
        '--max_len', type=int, default=1000, metavar='n',
        help="Longest sequence allowed. Other sequences will be zero-padded \n"
             "up to that length (but attention can't attend to padded "
             "timepoints).\nPlease beware that this value influences strongly "
             "the executing time and heaviness.\nAlso used with sinusoidal "
             "position embedding. [%(default)s]")
    gt.add_argument(
        '--nheads', type=int, default=8, metavar='n',
        help="Number of heads per layer. Could be different for each layer \n"
             "but we decided not to implement this possibility. [%(default)s]")
    gt.add_argument(
        '--dropout_rate', type=float, default=0.1, metavar='r',
        help="Dropout rate for all dropout layers. Again, could be different\n"
             "in every layers but that's not the choice we made.\n"
             "Needed in embedding, encoder and decoder. [%(default)s]")
    gt.add_argument(
        '--ffnn_hidden_size', type=int, default=None, metavar='n',
        help="Size of the feed-forward neural network (FFNN) layer in the \n"
             "encoder and decoder layers. The FFNN is composed of two linear\n"
             "layers. This is the size of the output of the first one. \n"
             "Default: data_embedding_size/2")
    gt.add_argument(
        '--activation', choices=['relu', 'gelu'], default='relu',
        metavar='key',
        help="Choice of activation function in the FFNN. One of 'relu' or \n"
             "'gelu'. [%(default)s]")
    gt.add_argument(
        '--norm_first', type=bool, default=False, metavar='True/False',
        help="If True, encoder and decoder layers will perform LayerNorm "
             "before \nother attention and feedforward operations, otherwise "
             "after.\n Torch default + in original paper: False. \nIn the "
             "tensor2tensor code, they suggest that learning is more robust "
             "when \npreprocessing each layer with the norm. Default: False.")
    gt.add_argument(
        "--start_from_copy_prev", action='store_true',
        help="If set, final_output = previous_dir + model_output.")

    g = p.add_argument_group("Neighborhood")
    AbstractTransformerModel.add_neighborhood_args_to_parser(g)

    g = p.add_argument_group("Output")
    AbstractTransformerModel.add_args_tracking_model(g)

    return gt


def add_tto_model_args(gt):
    gt.add_argument(
        '--n_layers_e', type=int, default=6,  metavar='n',
        help="Number of encoding layers. [%(default)s]")
    gt.add_argument(
        '--n_layers_d', type=int, default=6,  metavar='n',
        help="Number of decoding layers. [%(default)s]")


def add_ttst_model_args(gt):
    gt.add_argument(
        '--n_layers_d', type=int, default=14, metavar='n',
        help="Number of 'decoding' layers. Value in [3]; 14. "
             "Default: [%(default)s].\n"
             "[3]: https://arxiv.org/pdf/1905.06596.pdf")
    gt.add_argument(
        '--embedding_size_x', type=int, metavar='n',
        help="Embedding size for x. Default = 80%% of d_model.")


def perform_checks(args):
    # Deal with your optional parameters:
    if args.dropout_rate < 0 or args.dropout_rate > 1:
        raise ValueError('The dropout rate must be between 0 and 1.')

    if not args.ffnn_hidden_size:
        args.ffnn_hidden_size = int(args.d_model / 2)

    # Prepare args for the direction getter
    if not args.dg_dropout and args.dropout_rate:
        args.dg_dropout = args.dropout_rate
    dg_args = check_args_direction_getter(args)

    return args, dg_args
