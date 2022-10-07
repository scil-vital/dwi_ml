# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.projects.positional_encoding import (
    keys_to_positional_encodings)
from dwi_ml.models.projects.transformers import (AbstractTransformerModel,
                                                 OriginalTransformerModel,
                                                 TransformerSrcAndTgtModel)
from dwi_ml.models.utils.direction_getters import check_args_direction_getter


def add_abstract_model_args(p):
    """ Optional parameters for TransformingTractography"""
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

    g = p.add_argument_group("Previous directions")
    AbstractTransformerModel.add_args_model_with_pd(g)

    g = p.add_argument_group("Neighborhood")
    AbstractTransformerModel.add_neighborhood_args_to_parser(g)

    g = p.add_argument_group("Output")
    AbstractTransformerModel.add_args_tracking_model(g)

    return gt


def add_tto_model_args(gt):
    gt.add_argument(
        '--n_layers_e', type=int, default=6,
        help="Number of encoding layers. [%(default)s]")
    gt.add_argument(
        '--n_layers_d', type=int, default=6,
        help="Number of decoding layers. [%(default)s]")


def add_ttst_model_args(gt):
    gt.add_argument(
        '--n_layers_d', type=int, default=14,
        help="Number of 'decoding' layers. Value in [3]; 14. "
             "Default: [%(default)s].\n"
             "[3]: https://arxiv.org/pdf/1905.06596.pdf")


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


def prepare_tto_model(args, dg_args, sub_loggers_level):
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = OriginalTransformerModel(
            experiment_name=args.experiment_name, nb_features=args.nb_features,
            # Previous dirs:
            nb_previous_dirs=args.nb_previous_dirs,
            prev_dirs_embedding_size=args.prev_dirs_embedding_size,
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            normalize_prev_dirs=args.normalize_prev_dirs,
            # Concerning inputs:
            max_len=args.max_len,
            positional_encoding_key=args.position_encoding,
            embedding_key_x=args.data_embedding,
            embedding_key_t=args.target_embedding,
            # Torch's transformer parameters
            d_model=args.d_model, ffnn_hidden_size=args.ffnn_hidden_size,
            nheads=args.nheads, dropout_rate=args.dropout_rate,
            activation=args.activation, n_layers_e=args.n_layers_e,
            n_layers_d=args.n_layers_e,
            # Direction getter
            dg_key=args.dg_key, dg_args=dg_args,
            normalize_targets=args.normalize_targets,
            # Other
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            log_level=sub_loggers_level)

        logging.info("Transformer (original) model final parameters:" +
                     format_dict_to_str(model.params_for_json_prints))

    return model


def prepare_ttst_model(args, dg_args, sub_loggers_level):
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = TransformerSrcAndTgtModel(
            experiment_name=args.experiment_name, nb_features=args.nb_features,
            # Previous dirs:
            nb_previous_dirs=args.nb_previous_dirs,
            prev_dirs_embedding_size=args.prev_dirs_embedding_size,
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            normalize_prev_dirs=args.normalize_prev_dirs,
            # Concerning inputs:
            max_len=args.max_len,
            positional_encoding_key=args.position_encoding,
            embedding_key_x=args.data_embedding,
            embedding_key_t=args.target_embedding,
            # Torch's transformer parameters
            d_model=args.d_model, ffnn_hidden_size=args.ffnn_hidden_size,
            nheads=args.nheads, dropout_rate=args.dropout_rate,
            activation=args.activation, n_layers_d=args.n_layers_d,
            # Direction getter
            dg_key=args.dg_key, dg_args=dg_args,
            normalize_targets=args.normalize_targets,
            # Other
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            log_level=sub_loggers_level)

        logging.info("Transformer (src-tgt attention) model final "
                     "parameters:" +
                     format_dict_to_str(model.params_for_json_prints))

    return model
