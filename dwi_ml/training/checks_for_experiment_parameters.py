# -*- coding: utf-8 -*-

from os import path


# Various arg_parser equivalents.


def check_similar_to_none(var, var_name: str):
    """
    In yaml, None is understood with ~ or with 'null'. Here, we make sure
    that user didn't write something else close to null, which would be
    understood as a string.
    """
    if var == 'None':
        raise ValueError("You have set {} to None in yaml! Possible confusion, "
                         "stopping here. If you want to set a variable to None "
                         "with yaml, you should use ~ or "
                         "null.".format(var_name))
    if var == 'null':
        raise ValueError("You have set {} to 'null'. Possible confusion, "
                         "stopping here. Did you mean null without quotes?")


def check_required_was_given(var, var_name: str, required: bool):
    """
    With arg parser, we have mendatory and optional option. Not in yaml. To
    compensate, we add a 'required' arg in all check functions below. If
    required, an error is raised if value is None.

    A value can be None if user set it to None or if user simply didn't provide
    this parameter in the yaml file.
    """
    if required and var is None:
        raise ValueError("Choice for {} is required and should not be "
                         "None.".format(var_name))


def check_bool_instance(var: bool, var_name: str):
    # In case of bools, we do not allow the value None.
    if not isinstance(var, bool):
        raise ValueError('A bool value was expected for the variable {} but '
                         '{} was received'.format(var_name, var))


def check_float_instance(var: float, var_name: str):
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


# Checks for each parameters


def check_logging_level(level: str, required: bool = False):
    """
    Checks that level is one of ['error', 'warning', 'info', 'debug'] and
    returns level in upper case. If None and not required, default is warning.
    """
    check_similar_to_none(level, 'logging level')
    check_required_was_given(level, 'logging level', required)
    check_str_instance(level, 'level')

    if level is None:
        level = 'WARNING'
    else:
        level = level.upper()
        choices = ['ERROR', 'WARNING', 'INFO', 'DEBUG']
        if level not in choices:
            raise ValueError("Choice for logging level should be one of "
                             "['error', 'warning', 'info', 'debug'].")
    return level


def check_experiment_name(name: str, required: bool = False):
    check_similar_to_none(name, 'experiment name')
    check_required_was_given(name, 'experiment name', required)
    check_str_instance(name, 'name')

    return name


def check_hdf5_filename(filename: str, required: bool = False):
    check_similar_to_none(filename, 'hdf5 filename')
    check_required_was_given(filename, 'hdf5 filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The hdf5 database was not found! "
                         "({})".format(filename))
    return filename


def check_training_subjs_filename(filename: str, required: bool = False):
    check_similar_to_none(filename, 'training subjs filename')
    check_required_was_given(filename, 'training subjs filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The file training_subjs ({}) was not "
                         "found!".format(filename))
    return filename


def check_validation_subjs_filename(filename: str, required: bool = False):
    check_similar_to_none(filename, 'validation subjs filename')
    check_required_was_given(filename, 'validation subjs filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The file validation_subjs ({}) was not "
                         "found!".format(filename))
    return filename


def check_step_size(step_size: float, required: bool = False):
    check_similar_to_none(step_size, 'step_size')
    check_required_was_given(step_size, 'step_size', required)
    step_size = check_float_instance(step_size, 'step_size')

    if step_size is not None and step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using
        # scilpy.tracking.tools.resample_streamlines_step_size, a warning
        # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
        # that the value is suspicious. Not raising the same warnings here
        # as you may be wanting to test weird things to understand better
        # your model.
    return step_size


def check_add_noise(add_streamline_noise: bool):
    check_bool_instance(add_streamline_noise, 'add_streamline_noise')

    return add_streamline_noise


def check_split_ratio(split_ratio: float):
    # Not required. If not given, value is 0.
    check_similar_to_none(split_ratio, 'split_ratio')
    check_float_instance(split_ratio, 'split_ratio')

    if split_ratio is None:
        split_ratio = 0.0
    elif split_ratio > 1 or split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 and 1).")
    return split_ratio


def check_neighborhood(sphere_radius: float, grid_radius: int):
    """
    Returns: a tuple (str, [float | int]) = ('sphere' or 'grid', radius)
    """
    check_similar_to_none(sphere_radius, 'neighborhood sphere radius')
    check_similar_to_none(grid_radius, 'neighborhood grid radius')

    if sphere_radius and grid_radius:
        raise ValueError("You must choose maximum one neighborhood option "
                         "between sphere_radius and grid_radius!")
    if sphere_radius:
        return 'sphere', sphere_radius
    elif grid_radius:
        return 'grid', grid_radius
    else:
        return None, None


def check_previous_dir(num_previous_dirs: int):
    # Not required. If not given, value is 0.
    check_similar_to_none(num_previous_dirs, 'num_previous_dirs')
    check_int_instance(num_previous_dirs, 'num_previous_dirs')

    if num_previous_dirs is None:
        num_previous_dirs = 0
    return num_previous_dirs


def check_max_epochs(max_epochs: int, required: bool = False):
    check_similar_to_none(max_epochs, 'max_epochs')
    check_required_was_given(max_epochs, 'max_epochs', required)
    check_int_instance(max_epochs, 'max_epochs')

    return max_epochs


def check_patience(patience: int, required: bool = False):
    check_similar_to_none(patience, 'patience')
    check_required_was_given(patience, 'patience', required)
    check_int_instance(patience, 'patience')

    return patience


def check_batch_size(batch_size: int, required: bool = False):
    check_similar_to_none(batch_size, 'batch_size')
    check_required_was_given(batch_size, 'batch_size', required)
    check_int_instance(batch_size, 'batch_size')

    return batch_size


def check_volumes_per_batch(volumes_per_batch: int, required: bool = False):
    check_similar_to_none(volumes_per_batch, 'volumes_per_batch')
    check_required_was_given(volumes_per_batch, 'volumes_per_batch', required)
    check_int_instance(volumes_per_batch, 'volumes_per_batch')

    return volumes_per_batch


def check_cycles_per_volume(cycles_per_volume: int, volumes_per_batch: int):
    # Not required (if volumes_per_batch is None). If not given, value is 1.
    check_similar_to_none(cycles_per_volume, 'cycles_per_volume')
    check_int_instance(cycles_per_volume, 'cycles_per_volume')

    if cycles_per_volume is not None and volumes_per_batch is None:
        raise ValueError("You provided a value for cycles_per_volume_batch "
                         "but not for volumes_per_batch. That's not normal.")

    if cycles_per_volume is None:
        cycles_per_volume = 1

    return cycles_per_volume


def check_lazy(lazy: bool):
    check_bool_instance(lazy, 'lazy')

    return lazy


def check_cache_manager(cache_manager: bool, lazy: bool):
    check_bool_instance(cache_manager, 'cache_manager')

    if cache_manager and not lazy:
        raise ValueError("You set cache_manager to True but lazy is not used. "
                         "That's not normal.")
    return cache_manager


def check_use_gpu(use_gpu: bool):
    check_bool_instance(use_gpu, 'use_gpu')

    return use_gpu


def check_num_cpu_workers(num_workers: int, required: bool = False):
    check_similar_to_none(num_workers, 'num_workers')
    check_required_was_given(num_workers, 'num_workers', required)
    check_int_instance(num_workers, 'num_workers')

    return num_workers


def check_worker_interpolation(worker_interpolation: bool):
    check_bool_instance(worker_interpolation, 'worker_interpolation')

    return worker_interpolation


def check_taskman_managed(taskman_managed: bool):
    check_bool_instance(taskman_managed, 'taskman_managed')

    return taskman_managed


def check_seed(seed: int, required: bool = False):
    check_similar_to_none(seed, 'seed')
    check_required_was_given(seed, 'seed', required)
    check_int_instance(seed, 'seed')

    return seed


# Main check function


def check_all_experiment_parameters(conf):
    # Experiment:
    name = check_experiment_name(
        conf['experiment']['name'], required=False)

    # Dataset:
    hdf5_filename = check_hdf5_filename(
        conf['dataset']['hdf5_filename'], required=True)
    training_subjs_filename = check_training_subjs_filename(
        conf['dataset']['training_subjs_filename'], required=True)
    validation_subjs_filename = check_validation_subjs_filename(
        conf['dataset']['validation_subjs_filename'], required=True)

    # Preprocessing:
    step_size = check_step_size(
        conf['preprocessing']['step_size'], required=False)

    # Data augmentation:
    add_noise = check_add_noise(
        conf['data_augmentation']['add_noise'])
    split_ratio = check_split_ratio(
        conf['data_augmentation']['split_ratio'])

    # Input:
    neighborhood_type, neighborhood_radius = check_neighborhood(
        conf['input']['neighborhood']['sphere_radius'],
        conf['input']['neighborhood']['grid_radius'])
    num_previous_dirs = check_previous_dir(
        conf['input']['num_previous_dirs'])

    # Epochs:
    max_epochs = check_max_epochs(
        conf['training']['epochs']['max_epochs'], required=False)
    patience = check_patience(
        conf['training']['epochs']['patience'], required=False)
    batch_size = check_batch_size(
        conf['training']['batch']['size'], required=False)
    volumes_per_batch = check_volumes_per_batch(
        conf['training']['batch']['volumes_per_batch'], required=False)
    cycles_per_volume = check_cycles_per_volume(
        conf['training']['batch']['cycles_per_volume'], volumes_per_batch)

    # Memory:
    lazy = check_lazy(
        conf['memory']['lazy'])
    cache_manager = check_cache_manager(
        conf['memory']['cache_manager'], lazy)
    use_gpu = check_use_gpu(
        conf['memory']['use_gpu'])
    num_cpu_workers = check_num_cpu_workers(
        conf['memory']['num_cpu_workers'], required=False)
    worker_interpolation = check_worker_interpolation(
        conf['memory']['worker_interpolation'])
    taskman_managed = check_taskman_managed(
        conf['memory']['taskman_managed'])

    # Randomization:
    seed = check_seed(
        conf['randomization']['seed'])

    # Final args:
    arranged_conf = {
        'name': name,
        'hdf5_filename': hdf5_filename,
        'training_subjs_filename': training_subjs_filename,
        'validation_subjs_filename': validation_subjs_filename,
        'step_size': step_size,
        'add_noise': add_noise,
        'split_ratio': split_ratio,
        'neighborhood_type': neighborhood_type,
        'neighborhood_radius': neighborhood_radius,
        'num_previous_dirs': num_previous_dirs,
        'max_epochs': max_epochs,
        'patience': patience,
        'batch_size': batch_size,
        'volumes_per_batch': volumes_per_batch,
        'cycles_per_volume': cycles_per_volume,
        'lazy': lazy,
        'cache_manager': cache_manager,
        'use_gpu': use_gpu,
        'num_cpu_workers': num_cpu_workers,
        'worker_interpolation': worker_interpolation,
        'taskman_managed': taskman_managed,
        'seed': seed,
    }

    return arranged_conf
