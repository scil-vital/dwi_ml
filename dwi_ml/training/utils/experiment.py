# -*- coding: utf-8 -*-


def add_mandatory_args_training_experiment(p):
    p.add_argument(
        'experiments_path',
        help='Path where to save your experiment. \nComplete path will be '
             'experiments_path/experiment_name.')
    p.add_argument(
        'experiment_name',
        help='If given, name for the experiment.')
    p.add_argument(
        'hdf5_file',
        help='Path to the .hdf5 dataset. Should contain both your training \n'
             'and validation subjects.')
    p.add_argument(
        'input_group_name',
        help='Name of the input volume in the hdf5 dataset.')
    p.add_argument(
        'streamline_group_name',
        help="Name of the streamlines group in the hdf5 dataset.")


def add_memory_args_training_experiment(p):
    # Memory options both for the batch sampler and the trainer:
    m_g = p.add_argument_group("Memory options")
    m_g.add_argument(
        '--use_gpu', action='store_true',
        help="If set, as much as possible, computations will be done on GPU.\n"
             "In particular, we will avoid computations in the batch sampler,"
             "\nwhich works on the CPU.")
    m_g.add_argument(
        '--processes', type=int, default=0, metavar='n',
        help="Number of parallel CPU processes, when working on CPU. \n"
             "Default: 0 (no parallel processes).")
    m_g.add_argument(
        '--rng', type=int, default=1234,
        help="Random seed. [1234]")
