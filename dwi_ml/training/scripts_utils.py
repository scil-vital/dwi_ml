# -*- coding: utf-8 -*-

from os import path


def check_none(var, var_name: str):
    """
    In yaml, None is understood with ~ or with 'null'. Here, we make sure
    that user didn't write something else close to null, which would be
    understood as a string.
    """
    if var == 'None':
        raise ValueError("You have set {} to None! Possible confusion, "
                         "stopping here. If you want to set a variable to None "
                         "with yaml, you should use ~ or null.".format(var_name))
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


def check_bool_was_given(var: bool, var_name: str):
    if var is None:
        raise ValueError("You have set {} to None (in yaml, ~ or null). A bool "
                         "was expected (true or false).".format(var_name))



def check_logging_level(level: str, required: bool = False):
    """
    Checks that level is one of ['error', 'warning', 'info', 'debug'] and
    returns level in upper case. If None and not required, default is warning.
    """
    check_none(level, 'logging level')
    check_required_was_given(level, 'logging level', required)

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
    check_none(name, 'experiment name')
    check_required_was_given(name, 'experiment name', required)
    return name


def check_hdf5_filename(filename: str, required: bool = False):
    check_none(filename, 'hdf5 filename')
    check_required_was_given(filename, 'hdf5 filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The hdf5 database was not found! "
                         "({})".format(filename))
    return filename


def check_training_subjs_filename(filename: str, required: bool = False):
    check_none(filename, 'training subjs filename')
    check_required_was_given(filename, 'training subjs filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The file training_subjs ({}) was not "
                         "found!".format(filename))
    return filename


def check_validation_subjs_filename(filename: str, required: bool = False):
    check_none(filename, 'validation subjs filename')
    check_required_was_given(filename, 'validation subjs filename', required)

    if filename is not None and not path.exists(filename):
        raise ValueError("The file validation_subjs ({}) was not "
                         "found!".format(filename))
    return filename


def check_step_size(step_size: float, required: bool = False):
    check_none(step_size, 'step size')
    check_required_was_given(step_size, 'step_size', required)

    if step_size is not None and step_size <= 0:
        raise ValueError("Step size can't be 0 or less!")
        # Note. When using scilpy.tracking.tools.resample_streamlines_step_size,
        # a warning is shown if step_size < 0.1 or > np.max(sft.voxel_sizes),
        # saying that the value is suspicious. Not raising the same warnings
        # here as you may be wanting to test weird things to understand better
        # your model.
    return step_size


def check_add_noise(add_streamline_noise: bool):
    check_bool_was_given(add_streamline_noise, 'add_streamline_noise')
    return add_streamline_noise


def check_split_ratio(split_ratio: float):
    check_none(split_ratio, 'split_ratio')

    if split_ratio is None:
        split_ratio = 0
    elif split_ratio > 1 or split_ratio < 0:
        raise ValueError("split_ratio must be a ratio (i.e. between 0 and 1).")
    return split_ratio


def check_neighborhood(sphere_radius: float, grid_radius: int):
    """
    Returns: a tuple (type = 'sphere' or 'grid', radius)
    """
    check_none(sphere_radius, 'neighborhood sphere radius')
    check_none(grid_radius, 'neighborhood grid radius')

    if sphere_radius and grid_radius:
        raise ValueError("You must choose maximum one neighborhood option "
                         "between sphere_radius and grid_radius!")
    if sphere_radius:
        return 'sphere', sphere_radius
    elif grid_radius:
        return 'grid', grid_radius
    else:
        return None, None


def check_previous_dir(add_x_previous_dirs: int):
    # Not required. If not given, value is 0.
    check_none(add_x_previous_dirs, 'add_x_previous_dirs')

    if add_x_previous_dirs is None:
        add_x_previous_dirs = 0
    return add_x_previous_dirs


def check_max_epochs(max_epochs: int, required: bool = False):
    check_none(max_epochs, 'max_epochs')
    check_required_was_given(max_epochs, 'max_epochs', required)
    return max_epochs


def check_patience(patience: int, required: bool = False):
    check_none(patience, 'patience')
    check_required_was_given(patience, 'patience', required)
    return patience


def check_batch_size(batch_size: int, required: bool = False):
    check_none(batch_size, 'batch_size')
    check_required_was_given(batch_size, 'batch_size', required)
    return batch_size


def check_volumes_per_batch(volumes_per_batch: int, required: bool = False):
    check_none(volumes_per_batch, 'volumes_per_batch')
    check_required_was_given(volumes_per_batch, 'volumes_per_batch', required)
    return volumes_per_batch


def check_cycles_per_volume(cycles_per_volume: int, volumes_per_batch: int):
    # Not required (if volumes_per_batch is None). If not given, value is 1.
    check_none(cycles_per_volume, 'cycles_per_volume')

    if cycles_per_volume is not None and volumes_per_batch is None:
        raise ValueError("You provided a value for cycles_per_volume_batch "
                         "but not for volumes_per_batch. That's not normal.")

    if cycles_per_volume is None:
        cycles_per_volume = 1

    return cycles_per_volume


def check_lazy(lazy: bool):
    check_bool_was_given(lazy, 'lazy')
    return lazy


def check_cache_manager(cache_manager: bool, lazy: bool):
    check_bool_was_given(cache_manager, 'cache_manager')

    if cache_manager and not lazy:
        raise ValueError("You set cache_manager to True but lazy is not used. "
                         "That's not normal.")
    return cache_manager


def check_use_gpu(use_gpu: bool):
    check_bool_was_given(use_gpu, 'use_gpu')

    return use_gpu


def check_num_cpu_workers(num_workers: int, required: bool = False):
    check_none(num_workers, 'num_workers')
    check_required_was_given(num_workers, 'num_workers', required)
    return num_workers


def check_worker_interpolation(worker_interpolation: bool):
    check_bool_was_given(worker_interpolation, 'worker_interpolation')

    return worker_interpolation


def check_taskman_managed(taskman_managed: bool):
    check_bool_was_given(taskman_managed, 'taskman_managed')
    return taskman_managed


def check_seed(seed: int, required: bool = False):
    check_none(seed, 'seed')
    check_required_was_given(seed, 'seed', required)
    return seed