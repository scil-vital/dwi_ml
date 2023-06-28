# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from scilpy.io.utils import add_processes_arg


def add_args_to_parser(args, p):
    for arg, val in args.items():
        p.add_argument(arg, **val)


def add_logging_arg(p):
    p.add_argument(
        '--logging', default='WARNING', metavar='level',
        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help="Logging level. Note that, for readability, not all debug logs \n"
             "are printed in DEBUG mode, only the main ones.")


def add_resample_or_compress_arg(p: ArgumentParser):
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '--step_size', type=float, metavar='s',
        help="Step size to resample the data (in mm). Default: None")
    g.add_argument(
        '--compress', type=float, metavar='r', const=0.01, nargs='?',
        help="Compression ratio. Default: None. Default if set: 0.01.\n"
             "If neither step_size nor compress are chosen, streamlines "
             "will be kept \nas they are.")


def add_arg_existing_experiment_path(p: ArgumentParser):
    p.add_argument('experiment_path',
                   help='Path to the directory containing the experiment.\n'
                        '(Should contain a model subdir with a file \n'
                        'parameters.json and a file best_model_state.pkl.)')


def add_memory_args(p: ArgumentParser, add_lazy_options=False,
                    add_multiprocessing_option=True, add_rng=False):
    g = p.add_argument_group("Memory usage")

    # Multi-processing / GPU
    if add_multiprocessing_option:
        ram_options = g.add_mutually_exclusive_group()
        # Parallel processing or GPU processing
        add_processes_arg(ram_options)
        ram_options.add_argument(
            '--use_gpu', action='store_true',
            help="If set, use GPU for processing. Cannot be used \ntogether "
                 "with --processes.")
    else:
        p.add_argument('--use_gpu', action='store_true',
                       help="If set, use GPU for processing.")

    # RNG
    if add_rng:
        g.add_argument('--rng', type=int, default=1234,
                       help="Random seed. [1234]")

    # Lazy + cache size
    if add_lazy_options:
        g.add_argument(
            '--cache_size', type=int, metavar='s', default=1,
            help="Relevant only if lazy data is used. Size of the cache in "
                 "terms\n of length of the queue (i.e. number of volumes). \n"
                 "NOTE: Real cache size will actually be larger depending on "
                 "use;\nthe training, validation and testing sets each have "
                 "their cache. [1]")
        g.add_argument(
            '--lazy', action='store_true',
            help="If set, do not load all the dataset in memory at once. "
                 "Load \nonly what is needed for a batch.")

    return g
