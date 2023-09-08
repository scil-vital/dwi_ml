# -*- coding: utf-8 -*-
from argparse import ArgumentParser


def add_args_groups_to_parser(groups, p: ArgumentParser):
    for group_name, group_args in groups.items():
        g = p.add_argument_group(group_name + '\n:::::::::::::::::::::::::')
        for arg_name, parser_params in group_args.items():
            if 'mutually_exclusive_group' in arg_name:
                meg = g.add_mutually_exclusive_group()
                for sub_arg, sub_params in parser_params.items():
                    meg.add_argument(sub_arg, **sub_params)
            else:
                g.add_argument(arg_name, **parser_params)


def variable_names(arg_dict):
    keys = []
    for key in arg_dict.keys():
        if 'mutually_exclusive_group' in key:
            keys.extend(variable_names(arg_dict[key]))
        else:
            keys.append(key)
    return keys


def assert_no_same_args(list_of_args_dict, msg):
    """
    new_args: a dict of args: {str: parser_or_gui_dict}
    arg_groups: a dict of dicts of args: {str: {str: parser_or_gui_dict}}
    """

    all_arg_names = []
    for args_dict in list_of_args_dict:
        new_arg_names = variable_names(args_dict)
        all_arg_names.extend(new_arg_names)

    old_arg_names_unique = set(all_arg_names)

    assert len(old_arg_names_unique) == len(all_arg_names), msg


def get_logging_arg():
    return {
        '--logging': {
            'default': 'WARNING', 'metavar': 'level',
            'choices': ['ERROR', 'WARNING', 'INFO', 'DEBUG'],
            'help': "Logging level. Note that, for readability, not all "
                    "debug logs \nare printed in DEBUG mode, only the main "
                    "ones."}
    }


def get_overwrite_arg():
    return {'-f': {
        'dest': 'overwrite', 'action': 'store_true',
        'help': 'Force overwriting of the output files.'}
    }


def get_resample_or_compress_arg():
    args = {
        'mutually_exclusive_group_step_compress': {
            '--step_size': {
                'type': float, 'metavar': 's',
                'help': "Step size to resample the data (in mm). Default: None"},
            '--compress': {
                'type': float, 'metavar': 'r', 'const': 0.01, 'nargs': '?',
                'help': "Compression ratio. Default: None. Default if set: 0.01.\n"
                        "If neither step_size nor compress are chosen, "
                        "streamlines will be kept \nas they are."
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

        args.update({'mutually_exclusive_group_gpu_cpu': gpu_arg})
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
                'help':
                    "Relevant only if lazy data is used. Size of the cache in "
                    "terms\n of length of the queue (i.e. number of volumes). "
                    "\nNOTE: Real cache size will actually be larger depending on "
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
