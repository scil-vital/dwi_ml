# -*- coding: utf-8 -*-
import os
from argparse import ArgumentParser

from scilpy.io.utils import add_processes_arg


def add_resample_or_compress_arg(p: ArgumentParser):
    g = p.add_mutually_exclusive_group()
    g.add_argument(
        '--step_size', type=float, metavar='s',
        help="Step size to resample the data (in mm). Default: None")
    g.add_argument(
        '--compress', type=float, metavar='r', const=0.01, nargs='?',
        dest='compress_th',
        help="Compression ratio. Default: None. Default if set: 0.01.\n"
             "If neither step_size nor compress are chosen, streamlines "
             "will be kept \nas they are.")


def add_arg_existing_experiment_path(p: ArgumentParser):
    p.add_argument('experiment_path',
                   help='Path to the directory containing the experiment. '
                        '(Should contain a model subdir \nwith a file '
                        'parameters.json and a file best_model_state.pkl.)')
    p.add_argument('--use_latest_epoch', action='store_true',
                   help="If true, use model at latest epoch rather than "
                        "default (best model).")


def add_memory_args(p: ArgumentParser, add_lazy_options=False,
                    add_multiprocessing_option=True, add_rng=False):
    g = p.add_argument_group("Memory usage")

    # Multi-processing / GPU
    if add_multiprocessing_option:
        ram_options = g.add_mutually_exclusive_group()
        # Parallel processing or GPU processing
        ram_options.add_argument(
            '--processes', dest='nbr_processes', metavar='nb', type=int,
            default=1,
            help='Number of sub-processes to start for parallel processing. '
                 'Default: [%(default)s]')
        ram_options.add_argument(
            '--use_gpu', action='store_true',
            help="If set, use GPU for processing. Cannot be used together "
                 "with option --processes.")
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
                 "terms of length of the \nqueue (i.e. number of volumes). "
                 "NOTE: Real cache size will actually be larger \ndepending "
                 "on usage; the training, validation and testing sets each "
                 "have their \ncache. [1]")
        g.add_argument(
            '--lazy', action='store_true',
            help="If set, do not load all the dataset in memory at once. "
                 "Load only what is needed \nfor a batch.")

    return g


def verify_which_model_in_path(model_dir):
    model_type_txt = os.path.join(model_dir, 'model_type.txt')

    with open(model_type_txt, 'r') as txt_file:
        model_type = txt_file.readlines()

    model_type = model_type[0].replace('\n', '')
    return model_type
