# -*- coding: utf-8 -*-
import argparse

from dwi_ml.arg_utils import get_memory_args, assert_no_same_args, get_logging_arg
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.training.utils.batch_loaders import get_args_batch_loader
from dwi_ml.training.utils.batch_samplers import get_args_batch_sampler
from dwi_ml.training.utils.experiment import (
    get_mandatory_args_experiment_and_hdf5, get_comet_args)
from dwi_ml.training.utils.trainer import get_training_args


def get_all_args_groups_learn2track():
    trainer_args = get_training_args(add_a_tracking_validation_phase=True)
    trainer_args.update({
        '--clip_grad': {
            'type': float,
            'help': "Value to which the gradient norms to avoid exploding "
                    "gradients. \nDefault = None (not clipping)."}})

    not_grouped = get_logging_arg()
    # Step size / compress:
    not_grouped.update(Learn2TrackModel.get_args_main_model())

    args_input_data, args_rnn = get_l2t_args()
    groups = {
        'Experiment': get_mandatory_args_experiment_and_hdf5(),
        'Batch sampler': get_args_batch_sampler(),
        'Batch loader': get_args_batch_loader(),
        'Trainer': trainer_args,
        'Memory usage': get_memory_args(add_lazy_options=True, add_rng=True),
        'Comet.ml': get_comet_args(),
        'Learn2track: Main data': args_input_data,
        'Learn2track: Neighborhood':
            Learn2TrackModel.get_neighborhood_args(),
        'Learn2track: Main inputs embedding layer':
            Learn2TrackModel.get_args_input_embedding(),
        'Learn2track model: Previous directions embedding layer':
            Learn2TrackModel.get_args_model_with_pd(),
        'Learn2track: Main layer params': args_rnn,
        'Learn2track model: Direction Getter options':
            Learn2TrackModel.get_args_tracking_model(),
        'Other': not_grouped
    }

    assert_no_same_args(groups.values(),
                        'IMPLEMENTATION ERROR. Twice the same variable')

    return groups


def get_l2t_args():
    args_input_data = {
        'input_group_name': {
            'help': 'Name of the input volume in the hdf5 dataset.'},
        'streamline_group_name': {
            'help': "Name of the streamlines group in the hdf5 dataset."}
    }
    # Step size / compress:
    args_input_data.update(Learn2TrackModel.get_args_main_model())

    args_rnn = {
        '--rnn_key': {
            'choices': ['lstm', 'gru'], 'default': 'lstm',
            'help': "RNN version. Default: lstm."},
        '--rnn_layer_sizes': {
            'type': int, 'nargs': '+', 'metavar': 's', 'default': [100, 100],
            'help': "The output size after each layer (the real output "
                    "size depends on skip \nconnections). "
                    "Default = [100, 100]"},
        '--dropout': {
            'type': float, 'default': 0., 'metavar': 'r',
            'help': "Dropout ratio. If >0, add a dropout layer between "
                    "RNN layers."},
        '--use_skip_connection': {
            'action': 'store_true',
            'help': "Add skip connections. The pattern for skip connections "
                    "is as seen here:\n"
                    "https://arxiv.org/pdf/1308.0850v5.pdf"},
        '--use_layer_normalization': {
            'action': 'store_true',
            'help': "Add layer normalization. Explained here: \n"
                    "https://arxiv.org/pdf/1607.06450.pdf"},
        "--start_from_copy_prev": {
            'action': 'store_true',
            'help':
                "If set, final_output = previous_dir + model_output.\nThis "
                "can be used independantly from the other previous dirs "
                "options that define \nvalues to be concatenated to the "
                "input, and independently from skip connections \noutput, "
                "with concatenate values of each layer."}
    }
    return args_input_data, args_rnn
