# -*- coding: utf-8 -*-

""" Various arg_parser equivalents."""


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


def check_bool_instance(var: bool, var_name: str):
    # In case of bools, we do not allow the value None.
    if not isinstance(var, bool):
        raise ValueError('A bool value was expected for the variable {} but '
                         '{} was received'.format(var_name, var))


def check_float_instance(var: float, var_name: str) -> float:
    # We allow the value None.
    if var is not None and not isinstance(var, float):
        if isinstance(var, int):
            var = float(var)
        else:
            raise ValueError('A float value was expected for the variable {} '
                             'but {} was received.'.format(var_name, var))
    return var


def check_int_instance(var: int, var_name: str):
    # We allow the value None.
    if var is not None and not isinstance(var, int):
        raise ValueError("We expected a int for the variable {} but {} was "
                         "received".format(var_name, var))


def check_str_instance(var: str, var_name: str):
    # We allow the value None.
    if var is not None and not isinstance(var, str):
        raise ValueError('The variable {} should be a string, but {} was '
                         'received'.format(var_name, var))


def check_step_size(step_size: float) -> float:
    check_similar_to_none(step_size, 'step_size')
    step_size = check_float_instance(step_size, 'step_size')

    if step_size and step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.
    return step_size


def check_normalize_directions(normalize_directions: bool) -> bool:
    check_similar_to_none(normalize_directions, 'normalize_directions')
    return normalize_directions


def check_noise_size(noise_size: float) -> float:
    check_similar_to_none(noise_size, 'noise_size')
    check_float_instance(noise_size, 'noise_size')

    # Not really any maximum value for noise_size. Will be clipped based on
    # step_size during use. We allow value None.

    return noise_size


def check_noise_variability(noise_variability: float) -> float:
    check_similar_to_none(noise_variability, 'noise_variability')
    check_float_instance(noise_variability, 'noise_variability')

    # Not really any maximum value for noise_size. Will be clipped based on
    # step_size during use. We allow value None

    return noise_variability


def check_split_ratio(split_ratio: float):
    check_similar_to_none(split_ratio, 'split_ratio')
    check_float_instance(split_ratio, 'split_ratio')

    if split_ratio is None:
        split_ratio = 0.0
    elif split_ratio > 1 or split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 and 1).")
    return split_ratio


def check_reverse_ratio(reverse_ratio: float):
    check_similar_to_none(reverse_ratio, 'reverse_ratio')
    check_float_instance(reverse_ratio, 'reverse_ratio')

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
    check_similar_to_none(sphere_radius, 'neighborhood sphere radius')
    check_similar_to_none(grid_radius, 'neighborhood grid radius')

    if sphere_radius and grid_radius:
        raise ValueError("You must choose maximum one neighborhood option "
                         "between sphere_radius and grid_radius!")
    elif sphere_radius:
        return 'sphere', sphere_radius
    elif grid_radius:
        return 'grid', grid_radius
    else:
        return None, None


def check_previous_dir(num_previous_dirs: int):
    check_similar_to_none(num_previous_dirs, 'num_previous_dirs')
    check_int_instance(num_previous_dirs, 'num_previous_dirs')

    if num_previous_dirs is None:
        num_previous_dirs = 0

    return num_previous_dirs


def check_max_epochs(max_epochs: int):
    check_similar_to_none(max_epochs, 'max_epochs')
    check_int_instance(max_epochs, 'max_epochs')

    if max_epochs is None:
        raise Warning("Yaml parameters: the max_epochs should probably not be "
                      "None.")

    return max_epochs


def check_patience(patience: int):
    check_similar_to_none(patience, 'patience')
    check_int_instance(patience, 'patience')

    return patience


def check_batch_size(batch_size: int):
    check_similar_to_none(batch_size, 'batch_size')
    check_int_instance(batch_size, 'batch_size')

    if batch_size is None:
        raise Warning("Yaml parameter: batch size was set to None. This will "
                      "probably cause issues during sampling. Current batch "
                      "sampler implementation can not decide the maximum "
                      "batch size, it has to be given by the user.")

    return batch_size


def check_n_subjects_per_batch(n_subjects_per_batch: int):
    check_similar_to_none(n_subjects_per_batch, 'n_subjects_per_batch')
    check_int_instance(n_subjects_per_batch, 'n_subjects_per_batch')

    return n_subjects_per_batch


def check_cycles(cycles: int, n_subjects_per_batch: int):
    check_similar_to_none(cycles, 'cycles')
    check_int_instance(cycles, 'cycles')

    if cycles is not None and n_subjects_per_batch is None:
        raise ValueError("You provided a value for cycles but not for "
                         "n_subjects_per_batch. That's not normal.")

    if cycles is None:
        cycles = 1

    return cycles


def check_lazy(lazy: bool):
    check_bool_instance(lazy, 'lazy')

    return lazy


def check_cache_manager(cache_manager: bool, lazy: bool):
    check_bool_instance(cache_manager, 'cache_manager')

    if cache_manager and not lazy:
        raise ValueError("You set cache_manager to True but lazy is not used. "
                         "That's not normal.")
    return cache_manager


def check_avoid_cpu_computations(avoid_cpu_computations: bool):
    check_bool_instance(avoid_cpu_computations, 'avoid_cpu_computations')

    return avoid_cpu_computations


def check_num_cpu_workers(num_workers: int):
    check_similar_to_none(num_workers, 'num_workers')
    check_int_instance(num_workers, 'num_workers')

    return num_workers


def check_worker_interpolation(worker_interpolation: bool):
    check_bool_instance(worker_interpolation, 'worker_interpolation')

    return worker_interpolation


def check_taskman_managed(taskman_managed: bool):
    check_bool_instance(taskman_managed, 'taskman_managed')

    return taskman_managed


def check_rng_seed(seed: int) -> int:
    check_similar_to_none(seed, 'seed')
    check_int_instance(seed, 'seed')

    if seed is None:
        raise Warning('Yaml parameters: Seed value should probably be given '
                      'instead of set to None.')

    return seed


# Main check function
def check_all_experiment_parameters(conf: dict):
    all_params = dict()

    # Preprocessing:
    all_params['step_size'] = check_step_size(
        conf['preprocessing']['step_size'])
    all_params['normalize_directions'] = check_normalize_directions(
        conf['preprocessing']['normalize_directions'])

    # Data augmentation:
    all_params['noise_size'] = check_noise_size(
        conf['data_augmentation']['noise_size'])
    all_params['noise_variability'] = check_noise_variability(
        conf['data_augmentation']['noise_variability'])
    all_params['split_ratio'] = check_split_ratio(
        conf['data_augmentation']['split_ratio'])
    all_params['reverse_ratio'] = check_reverse_ratio(
        conf['data_augmentation']['reverse_ratio'])

    # Input:
    (n_type, n_radius) = check_neighborhood(
        conf['input']['neighborhood']['sphere_radius'],
        conf['input']['neighborhood']['grid_radius'])
    all_params['neighborhood_type'] = n_type
    all_params['neighborhood_radius'] = n_radius
    all_params['num_previous_dirs'] = check_previous_dir(
        conf['input']['num_previous_dirs'])

    # Epochs:
    all_params['max_epochs'] = check_max_epochs(
        conf['training']['epochs']['max_epochs'])
    all_params['patience'] = check_patience(
        conf['training']['epochs']['patience'])
    all_params['batch_size'] = check_batch_size(
        conf['training']['batch']['size'])
    n_subjects_per_batch = check_n_subjects_per_batch(
        conf['training']['batch']['n_subjects_per_batch'])
    all_params['n_subjects_per_batch'] = n_subjects_per_batch
    all_params['cycles'] = check_cycles(
        conf['training']['batch']['cycles'], n_subjects_per_batch)

    # Memory:
    lazy = check_lazy(
        conf['memory']['lazy'])
    all_params['lazy'] = lazy
    all_params['cache_manager'] = check_cache_manager(
        conf['memory']['cache_manager'], lazy)
    all_params['avoid_cpu_computations'] = check_avoid_cpu_computations(
        conf['memory']['avoid_cpu_computations'])
    all_params['num_cpu_workers'] = check_num_cpu_workers(
        conf['memory']['num_cpu_workers'])
    all_params['worker_interpolation'] = check_worker_interpolation(
        conf['memory']['worker_interpolation'])
    all_params['taskman_managed'] = check_taskman_managed(
        conf['memory']['taskman_managed'])

    # Randomization:
    all_params['rng'] = check_rng_seed(
        conf['randomization']['rng'])

    return all_params
