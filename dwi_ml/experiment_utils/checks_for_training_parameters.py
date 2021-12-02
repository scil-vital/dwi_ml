# -*- coding: utf-8 -*-

""" Various arg_parser equivalents."""
import logging
from typing import List

import torch
from dwi_ml.experiment_utils.prints import format_dict_to_str


def check_similar_to_none(var, var_name: str):
    """
    In yaml, None is understood with ~ or with 'null'. Here, we make sure
    that user didn't write something else close to null, which would be
    understood as a string.
    """
    if var == 'None':
        raise ValueError("You have set {} to None in yaml! Possible "
                         "confusion, stopping here. If you want to set a "
                         "variable to None with yaml, you should use ~ or "
                         "null.".format(var_name))
    if var == 'null':
        raise ValueError("You have set {} to 'null'. Possible confusion, "
                         "stopping here. Did you mean null without quotes?")


def check_bool_instance_not_none(var: bool, var_name: str):
    # In case of bools, we do **not** allow the value None.
    if not isinstance(var, bool):
        raise ValueError('A bool value was expected for the variable {} but '
                         '{} was received'.format(var_name, var))
    return var


def check_float_or_none_instance(var: float, var_name: str, fix_none=None):
    check_similar_to_none(var, 'var_name')

    if var is not None and not isinstance(var, float):
        if isinstance(var, int):
            var = float(var)
        else:
            raise ValueError('A float value was expected for the variable {} '
                             'but {} was received.'.format(var_name, var))
    if var is None and fix_none:
        var = fix_none

    return var


def check_int_or_none_instance(var: int, var_name: str, fix_none=None):
    check_similar_to_none(var, 'var_name')

    if var is not None and not isinstance(var, int):
        raise ValueError("We expected a int for the variable {} but {} was "
                         "received".format(var_name, var))
    if var is None and fix_none:
        var = fix_none

    return var


def check_str_or_none_instance(var: str, var_name: str):
    check_similar_to_none(var, 'var_name')

    if var is not None and not isinstance(var, str):
        raise ValueError('The variable {} should be a string, but {} was '
                         'received'.format(var_name, var))
    return var


def check_step_size(step_size: float) -> float:
    step_size = check_float_or_none_instance(step_size, 'step_size')

    if step_size and step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.
    return step_size


def check_split_ratio(split_ratio: float):
    check_float_or_none_instance(split_ratio, 'split_ratio')

    if split_ratio is None:
        split_ratio = 0.0
    elif split_ratio > 1 or split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 and 1).")
    return split_ratio


def check_reverse_ratio(reverse_ratio: float):
    check_float_or_none_instance(reverse_ratio, 'reverse_ratio')

    if reverse_ratio is None:
        reverse_ratio = 0.0
    elif reverse_ratio > 1 or reverse_ratio < 0:
        raise ValueError("reverse_ratio must be a ratio (i.e. between 0 and "
                         "1).")
    return reverse_ratio


def check_neighborhood(sphere_radius: float, grid_radius: int):
    """
    Returns: a tuple (str, [float | int]) = ('sphere' or 'grid', radius)
    """
    if isinstance(sphere_radius, List):
        check_float_or_none_instance(sphere_radius[0],
                                     'first neighborhood sphere radius')
    else:
        check_float_or_none_instance(sphere_radius,
                                     'neighborhood sphere radius')
    check_int_or_none_instance(grid_radius, 'neighborhood grid radius')

    if sphere_radius and grid_radius:
        raise ValueError("You must choose maximum one neighborhood option "
                         "between sphere_radius and grid_radius!")
    elif sphere_radius:
        return 'axes', sphere_radius
    elif grid_radius:
        return 'grid', grid_radius
    else:
        return None, None


def check_epochs(max_epochs: int, max_batches_per_epoch: int):
    check_int_or_none_instance(max_epochs, 'max_epochs')
    check_int_or_none_instance(max_batches_per_epoch, 'max_batches_per_epoch')

    if max_epochs is None:
        logging.warning("Yaml parameters: the max_epochs should probably not "
                        "be None.")
    if max_batches_per_epoch is None:
        logging.warning("Yaml parameters: the max_batches_per_epoch should "
                        "probably not be None.")

    return max_epochs, max_batches_per_epoch


def check_chunk_size(chunk_size: int):
    check_int_or_none_instance(chunk_size, 'chunk_size')

    if chunk_size is None:
        logging.warning("Yaml parameter: chunk size was set to None. This "
                        "will probably cause issues during sampling. Current "
                        "batch sampler implementation can not decide the "
                        "the number of streamlines to load for each chunk,"
                        "it has to be given by the user.")

    return chunk_size


def check_batch_size(batch_size: int):
    check_int_or_none_instance(batch_size, 'batch_size')

    if batch_size is None:
        logging.warning("Yaml parameter: batch size was set to None. This "
                        "will probably cause issues during sampling. Current "
                        "batch sampler implementation can not decide the "
                        "maximum batch size, it has to be given by the user.")

    return batch_size


def check_cycles(cycles: int, nb_subjects_per_batch: int):
    check_int_or_none_instance(cycles, 'cycles')

    if cycles is not None and nb_subjects_per_batch is None:
        raise ValueError("You provided a value for cycles but not for "
                         "nb_subjects_per_batch. That's not normal.")

    if cycles is None:
        cycles = 1

    return cycles


def check_cache_manager(cache_size: int, lazy: bool):
    check_int_or_none_instance(cache_size, 'cache_manager')

    if cache_size and not lazy:
        logging.warning("You set cache_manager to True but lazy is not used. "
                        "Value will be ignored.")
    if cache_size is None and lazy:
        raise ValueError("Cache size cannot be 0 with lazy data!")
    return cache_size


def check_gpu(use_gpu: bool):
    check_bool_instance_not_none(use_gpu, 'use_gpu')
    if use_gpu and not torch.cuda.is_available():
        logging.warning("User provided option 'use_gpu' but cuda is not "
                        "available. Using CPU!")
        use_gpu = False
    return use_gpu


def check_rng_seed(seed: int) -> int:
    check_int_or_none_instance(seed, 'seed')

    if seed is None:
        logging.warning('Yaml parameters: Seed value should probably be given '
                        'instead of set to None.')

    return seed


# Main check function
def check_all_experiment_parameters(conf: dict):
    logging.debug("Parameters loaded from the yaml file: " +
                  format_dict_to_str(conf))

    # Sampling
    s = conf['sampling']  # to shorten vars
    nb_subjects_per_batch = check_int_or_none_instance(
        s['batch']['nb_subjects_per_batch'],
        'nb_subjects_per_batch')
    sampling_params = {
        # batch
        'chunk_size': check_chunk_size(s['batch']['chunk_size']),
        'max_batch_size': check_batch_size(s['batch']['max_batch_size']),
        'nb_subjects_per_batch': nb_subjects_per_batch,
        'cycles': check_cycles(s['batch']['cycles'], nb_subjects_per_batch),

        # processing
        'step_size': check_step_size(
            s['streamlines']['processing']['step_size']),
        'compress': check_bool_instance_not_none(
            s['streamlines']['processing']['compress'],
            'compress'),
        'normalize_directions': check_bool_instance_not_none(
            s['streamlines']['processing']['normalize_directions'],
            'normalize_directions'),

        # data_augmentation
        'noise_gaussian_size': check_float_or_none_instance(
            s['streamlines']['data_augmentation']['noise_size'], 'noise_size',
            fix_none=0.),
        'noise_gaussian_variability': check_float_or_none_instance(
            s['streamlines']['data_augmentation']['noise_variability'],
            'noise_variability', fix_none=0.),
        'split_ratio': check_split_ratio(
            s['streamlines']['data_augmentation']['split_ratio']),
        'reverse_ratio': check_reverse_ratio(
            s['streamlines']['data_augmentation']['reverse_ratio'])}

    # Training:
    e, b = check_epochs(conf['training']['epochs']['max_epochs'],
                        conf['training']['epochs']['max_batches_per_epoch'])
    training_params = {
        'learning_rate': check_float_or_none_instance(
            conf['training']['learning_rate'], 'learning_rate',
            fix_none=0.001),
        'weight_decay': check_float_or_none_instance(
            conf['training']['weight_decay'], 'lweight_decay', fix_none=0.01),

        # epochs:
        'max_epochs': e,
        'patience': check_int_or_none_instance(
            conf['training']['epochs']['patience'], 'patience'),
        'max_batches_per_epoch': b,
        }

    # Model:
    (n_type, n_radius) = check_neighborhood(
        conf['model']['neighborhood']['sphere_radius'],
        conf['model']['neighborhood']['grid_radius'])
    model_params = {
        'nb_previous_dirs': check_int_or_none_instance(
            conf['model']['previous_dirs']['nb_previous_dirs'],
            'nb_previous_dirs', fix_none=0),
        'neighborhood_type': n_type,
        'neighborhood_radius': n_radius
    }

    # Memory:
    lazy = check_bool_instance_not_none(
        conf['memory']['lazy'], 'lazy')
    memory_params = {
        'lazy': lazy,
        'cache_size': check_cache_manager(conf['memory']['cache_size'], lazy),
        'use_gpu': check_gpu(conf['memory']['use_gpu']),
        'nb_cpu_workers': check_int_or_none_instance(
            conf['memory']['nb_cpu_workers'], 'nb_cpu_workers'),
        'worker_interpolation': check_bool_instance_not_none(
            conf['memory']['worker_interpolation'], 'worker_interpolation'),
        'taskman_managed': check_bool_instance_not_none(
            conf['memory']['taskman_managed'], 'taskman_managed')}

    randomization = {
        'rng': check_rng_seed(conf['randomization']['rng'])}

    return (sampling_params, training_params, model_params, memory_params,
            randomization)
