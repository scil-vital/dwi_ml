# -*- coding: utf-8 -*-
import argparse


def parse_args_train_model():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Path where to save your experiment. Complete path '
                        'will be experiment_path/experiment_name.')
    p.add_argument('--input_group',
                   help='Name of the input group. If a checkpoint exists, '
                        'this information is already contained in the '
                        'checkpoint and is not necessary.')
    p.add_argument('--target_group',
                   help='Name of the target streamline group. If a checkpoint '
                        'exists, this information is already contained in the '
                        'checkpoint and is not necessary.')
    p.add_argument('--hdf5_file',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects. If a '
                        'checkpoint exists, this information is already '
                        'contained in the checkpoint and is not necessary.')
    p.add_argument('--parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example. If a checkpoint exists, this information '
                        'is already contained in the checkpoint and is not '
                        'necessary.')
    p.add_argument('--experiment_name',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give based on time of day.')
    p.add_argument('--override_checkpoint_patience', type=int,
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment to continue if the allowed '
                        'number of bad epochs has been previously reached.')
    p.add_argument('--override_checkpoint_max_epochs', type=int,
                   help='If a checkpoint exists, max_epochs can be increased '
                        'to allow experiment to continue if the allowed '
                        'number of epochs has been previously reached.')
    p.add_argument('--logging', choices=['error', 'warning', 'info', 'debug'],
                   help="Logging level. One of ['error', 'warning', 'info', "
                        "'debug']. Default: Info.")
    p.add_argument('--comet_workspace',
                   help='Your comet workspace. If not set, comet.ml will not '
                        'be used. See our docs/Getting Started for more '
                        'information on comet and its API key.')
    p.add_argument('--comet_project',
                   help='Send your experiment to a specific comet.ml project. '
                        'If not set, it will be sent to Uncategorized '
                        'Experiments.')

    arguments = p.parse_args()

    return arguments
