# -*- coding: utf-8 -*-
from argparse import ArgumentParser

from scilpy.io.utils import add_processes_arg


def add_args_groups_to_parser(groups, p: ArgumentParser):
    for group_name, group_args in groups.items():
        g = p.add_argument_group(groups[group_name])
        for arg, val in group_args.items():
            if arg == 'mutually_exclusive_group':
                meg = g.add_mutually_exclusive_group()
                for sub_arg, sub_val in val.items():
                    meg.add_argument(sub_arg, **sub_val)
            else:
                g.add_argument(arg, **val)


def add_logging_arg(p):
    p.add_argument(
        '--logging', default='WARNING', metavar='level',
        choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
        help="Logging level. Note that, for readability, not all debug logs \n"
             "are printed in DEBUG mode, only the main ones.")


def get_resample_or_compress_arg():
    args = {
        'mutually_exclusive_group': {
            '--step_size': {
                'type': float, 'metavar': 's',
                'help': "Step size to resample the data (in mm). Default: None"},
            '--compress': {
                'type': float, 'metavar': 'r', 'const': 0.01, 'nargs': '?',
                'help': "Compression ratio. Default: None. Default if set: 0.01.\n"
                        "If neither step_size nor compress are chosen, streamlines "
                        "will be kept \nas they are."
            }
        }
    }
    return args


def add_arg_existing_experiment_path(p: ArgumentParser):
    p.add_argument('experiment_path',
                   help='Path to the directory containing the experiment.\n'
                        '(Should contain a model subdir with a file \n'
                        'parameters.json and a file best_model_state.pkl.)')


def get_memory_args(add_lazy_options=False,
                    add_multiprocessing_option=True, add_rng=False):
    args = {}

    # Multi-processing / GPU
    gpu_arg = {
        '--use_gpu': {
            'action': 'store_true',
            'help': "If set, use GPU for processing. Cannot be used "
                    "together with --processes."}}
    if add_multiprocessing_option:
        gpu_arg.update({
            '--nbr_processes': {
                'metavar': 'n', 'type': int, 'default': 1,
                'help': "Number of sub-processes to start. Default: [%(default)s]"}
        })

        args.update({'mutually_exclusive_group': gpu_arg})
    else:
        args.update(gpu_arg)

    # RNG
    if add_rng:
        args.update({
            '--rng': {
                'type': int, 'default': 1234,
                'help': "Random seed. [1234]"
            }
        })

    # Lazy + cache size
    if add_lazy_options:
        args.update({
            '--cache_size': {
                'metavar': 'n', 'type': int, 'default': 1,
                'help': "Relevant only if lazy data is used. Size of the cache in "
                        "terms\n of length of the queue (i.e. number of volumes). \n"
                        "NOTE: Real cache size will actually be larger depending on "
                        "use;\nthe training, validation and testing sets each have "
                        "their cache. [1]"
            },
            '--lazy': {
                'action': 'store_true',
                'help': "If set, do not load all the dataset in memory at once. "
                        "Load \nonly what is needed for a batch."
            }
        })

    return args
