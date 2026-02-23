# -*- coding: utf-8 -*-
import argparse

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel


def add_model_args(p: argparse.ArgumentParser):
    # Mandatory args
    p.add_argument(
        'input_group_name',
        help='Name of the input volume in the hdf5 dataset.')
    p.add_argument(
        'streamline_group_name',
        help="Name of the streamlines group in the hdf5 dataset.")

    # Optional args

    # Step_size / compress
    Learn2TrackModel.add_args_main_model(p)
    
    prev_dirs_g = p.add_argument_group(
        "Learn2track model: Previous directions embedding layer")
    Learn2TrackModel.add_args_model_with_pd(prev_dirs_g)

    inputs_g = p.add_argument_group(
        "Learn2track model: Main inputs embedding layer")
    Learn2TrackModel.add_neighborhood_args_to_parser(inputs_g)
    Learn2TrackModel.add_args_input_embedding(inputs_g)

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

    g = p.add_argument_group("Learn2track model: Direction Getter options")
    Learn2TrackModel.add_args_tracking_model(g)
