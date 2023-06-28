# -*- coding: utf-8 -*-


def get_mandatory_args_experiment_and_hdf5():
    args = {
        'experiments_path': {
            'help': "Path where to save your experiment. \n"
                    "Complete path will be experiments_path/experiment_name."},
        'experiment_name': {
            'help': 'Name for the experiment.'},
        'hdf5_file': {
            'help': 'Path to the .hdf5 dataset. Should contain both your '
                    'training and \nvalidation subjects.'},
    }
    return args


def add_args_resuming_experiment(p):
    p.add_argument('experiments_path',
                   help='Path from where to load your experiment, and where to'
                        'save new results.\nComplete path will be '
                        'experiments_path/experiment_name.')
    p.add_argument('experiment_name',
                   help='Name of the experiment.')
    p.add_argument('--hdf5_file',
                   help='Path to the .hdf5 dataset. If not given, uses the '
                        'same file as in the first portion of training.')

    p.add_argument('--new_patience', type=int, metavar='new_p',
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of bad epochs has been previously reached.')
    p.add_argument('--new_max_epochs', type=int,
                   metavar='new_max',
                   help='If a checkpoint exists, max_epochs can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of epochs has been previously reached.')

    return p
