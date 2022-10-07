# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel


def add_model_args(p: argparse.ArgumentParser):
    prev_dirs_g = p.add_argument_group(
        "Learn2track model: Previous directions embedding layer")
    Learn2TrackModel.add_args_model_with_pd(prev_dirs_g)

    inputs_g = p.add_argument_group(
        "Learn2track model: Main inputs embedding layer")
    Learn2TrackModel.add_neighborhood_args_to_parser(inputs_g)
    inputs_g.add_argument(
        '--input_embedding_key', choices=keys_to_embeddings.keys(),
        default='no_embedding',
        help="Type of model for the inputs embedding layer.\n"
             "Default: no_embedding (identity model).")
    em_size = inputs_g.add_mutually_exclusive_group()
    em_size.add_argument(
        '--input_embedding_size', type=int, metavar='s',
        help="Size of the output after passing the previous dirs through the "
             "embedding layer. \nDefault: embedding_size=input_size.")
    em_size.add_argument(
        '--input_embedding_size_ratio', type=float, metavar='r',
        help="Size of the output after passing the previous dirs through the "
             "embedding layer. \nThe inputs size (i.e. number of features per "
             "voxel) will be verified when \nloading the data, and the "
             "embedding size wil be output_size_ratio*nb_features.\n"
             "Default: 1.")

    rnn_g = p.add_argument_group("Learn2track model: RNN layer")
    rnn_g.add_argument(
        '--rnn_key', choices=['lstm', 'gru'], default='lstm',
        help="RNN version. Default: lstm.")
    rnn_g.add_argument(
        '--rnn_layer_sizes', type=int, nargs='+', metavar='s',
        default=[100, 100],
        help="The output size after each layer (the real output size depends "
             "on skip \nconnections). Default = [100, 100]")
    rnn_g.add_argument(
        '--dropout', type=float, default=0., metavar='r',
        help="Dropout ratio. If >0, add a dropout layer between RNN layers.")
    rnn_g.add_argument(
        '--use_skip_connection', action='store_true',
        help="Add skip connections. The pattern for skip connections is as "
             "seen here:\n"
             "https://arxiv.org/pdf/1308.0850v5.pdf")
    rnn_g.add_argument(
        '--use_layer_normalization', action='store_true',
        help="Add layer normalization. Explained here: \n"
             "https://arxiv.org/pdf/1607.06450.pdf")

    g = p.add_argument_group("Learn2track model: others")
    Learn2TrackModel.add_args_tracking_model(g)


def prepare_model(args, dg_args, log_level):
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        # INPUTS: verifying args
        model = Learn2TrackModel(
            experiment_name=args.experiment_name,
            # PREVIOUS DIRS
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            prev_dirs_embedding_size=args.prev_dirs_embedding_size,
            nb_previous_dirs=args.nb_previous_dirs,
            normalize_prev_dirs=args.normalize_prev_dirs,
            # INPUTS
            input_embedding_key=args.input_embedding_key,
            input_embedding_size=args.input_embedding_size,
            input_embedding_size_ratio=args.input_embedding_size_ratio,
            nb_features=args.nb_features,
            # RNN
            rnn_key=args.rnn_key, rnn_layer_sizes=args.rnn_layer_sizes,
            dropout=args.dropout,
            use_layer_normalization=args.use_layer_normalization,
            use_skip_connection=args.use_skip_connection,
            # TRACKING MODEL
            dg_key=args.dg_key, dg_args=dg_args,
            normalize_targets=args.normalize_targets,
            # Other
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            log_level=log_level)

        # logging.info("Learn2track model user-defined parameters: \n" +
        #              format_dict_to_str(model.params) + '\n')
        logging.info("Learn2track model final parameters:" +
                     format_dict_to_str(model.params_for_json_prints))
    return model
