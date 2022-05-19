# -*- coding: utf-8 -*-


def add_mandatory_args_training_experiment(p):
    p.add_argument(
        'experiments_path',
        help='Path where to save your experiment. \nComplete path will be '
             'experiments_path/experiment_name.')
    p.add_argument(
        'experiment_name',
        help='If given, name for the experiment. Else, model will decide the '
             'name to \ngive based on time of day.')
    p.add_argument(
        'hdf5_file',
        help='Path to the .hdf5 dataset. Should contain both your training '
             'and \nvalidation subjects.')
    p.add_argument(
        'input_group_name',
        help='Name of the input volume in the hdf5 dataset.')
    p.add_argument(
        'streamline_group_name',
        help="Name of the streamlines group in the hdf5 dataset.")


def add_printing_args_training_experiment(p):
    p.add_argument(
        '--logging', dest='logging_choice', default='WARNING',
        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help="Logging level. Note that, for readability, not all debug logs "
             "are printed in DEBUG mode.")
    p.add_argument(
        '--taskman_managed', action='store_true',
        help="If set, instead of printing progression, print taskman-relevant "
             "data.")


def add_memory_args_training_experiment(p):
    # Memory options both for the batch sampler and the trainer:
    m_g = p.add_argument_group("Memory options :")
    m_g.add_argument(
        '--use_gpu', action='store_true',
        help="If set, avoids computing and interpolating the inputs (and "
             "their neighborhood) \ndirectly in the batch sampler (which is "
             "computed on CPU). Will be computed later on GPU, \nin the "
             "trainer.")
    m_g.add_argument(
        '--processes', type=int, default=0,
        help="Number of parallel CPU processes, when working on CPU.")
    m_g.add_argument(
        '--rng', type=int, default=1234,
        help="Random seed. Default=1234.")
