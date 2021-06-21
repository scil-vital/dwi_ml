#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# Remove or add parameters to fit your needs. You should change your yaml file
# accordingly.
# Change DWIMLAbstractSequences for an implementation of your own model.
# It should be a child of this abstract class.
#
# Choose the batch sampler that fits your model
# So far, we have implemented one version, where
#     x: input = volume group named "input"
#     y: target = the streamlines
############################

""" Train a model for my favorite experiment"""

import argparse
import logging
from os import path
import yaml

import numpy as np
import torch

from dwi_ml.model.batch_samplers import (TrainingBatchSamplerOneInputVolume,
                                         LazyTrainingBatchSamplerOneInputVolume)
from dwi_ml.training.checks_for_experiment_parameters import (
    check_all_experiment_parameters, check_logging_level)
from dwi_ml.training.trainer_abstract import DWIMLTrainerAbstractSequences


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects.')
    p.add_argument('parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example.')
    p.add_argument('--experiment_name', default=None,
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give')

    arguments = p.parse_args()

    return arguments


def init_datasets(hdf5_filename, name,
                  lazy, cache_manager, volumes_per_batch, seed,
                  add_streamline_noise, step_size, neighborhood_dist_mm,
                  streamlines_cut_ratio, add_previous_dir,
                  worker_interpolation, dataset_device, taskman_managed):
    # Choose the class used to represent your data
    # (You may change these for your own implementations or for a child class)
    other_kw_args = {}
    if lazy:
        dataset_cls = LazyTrainingBatchSamplerOneInputVolume
        if cache_manager:
            other_kw_args['cache_size'] = volumes_per_batch
    else:
        dataset_cls = TrainingBatchSamplerOneInputVolume

    rng = np.random.RandomState(seed)
    dataset = dataset_cls(
        hdf5_filename, rng, name=name,
        use_streamline_noise=add_streamline_noise, step_size=step_size,
        neighborhood_dist_mm=neighborhood_dist_mm,
        streamlines_cut_ratio=streamlines_cut_ratio,
        add_previous_dir=add_previous_dir,
        do_interpolation=worker_interpolation, device=dataset_device,
        taskman_managed=taskman_managed, **other_kw_args)

    return dataset


def main():
    args = parse_args()

    # Check that all files exist
    if not path.exists(args.training_subjs_filename):
        raise ValueError("The training subjects list ({}) was not found!"
                         .format(args.training_subjs_filename))
    if args.validation_subjs_filename and \
            not path.exists(args.validation_subjs_filename):
        raise ValueError("The validation subjects list ({}) was not found!"
                         .format(args.validation_subjs_filename))
    if not path.exists(args.hdf5_filename):
        raise ValueError("The validation subjects list ({}) was not found!"
                         .format(args.hdf5_filename))
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("Yaml file not found: {}"
                                .format(args.parameters_filename))

    # Load parameters
    with open(args.parameters_filename) as f:
        yaml_parameters = yaml.safe_load(f.read())

    # Initialize logger
    logging_level = check_logging_level(yaml_parameters['logging']['level'],
                                        required=False)
    logging.basicConfig(level=logging_level)
    logging.info(yaml_parameters)

    # Perform checks
    # We have decided to use yaml for a more rigorous way to store parameters,
    # compared, say, to bash. However, no argparser is used so we need to
    # make our own checks.
    (step_size, add_noise, split_ratio, neighborhood_type, neighborhood_radius,
     num_previous_dirs, max_epochs, patience, batch_size, volumes_per_batch,
     cycles_per_volume, lazy, cache_manager, use_gpu, num_cpu_workers,
     worker_interpolation, taskman_managed,
     seed) = check_all_experiment_parameters(yaml_parameters)

    # Set device
    if use_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Instantiate dataset class
    # dataset contains both the traning set and validation set data.
    dataset = init_datasets(args.hdf5_filename, args.experiment_name, lazy,
                            cache_manager=cache_manager,
                            volumes_per_batch=volumes_per_batch, seed=seed,
                            add_streamline_noise=add_noise,
                            step_size=step_size,
                            neighborhood_dist_mm=neighborhood_radius,
                            streamlines_cut_ratio=split_ratio,
                            add_previous_dir=num_previous_dirs,
                            worker_interpolation=worker_interpolation,
                            dataset_device=device,
                            taskman_managed=taskman_managed)

    # Instantiate trainer class
    # (Change DWIMLAbstractSequences for your class.)
    # Then load dataset, build model, train and save
    experiment = DWIMLTrainerAbstractSequences(
        dataset, args.name, args.hdf5_filename,
        max_epochs=max_epochs, patience=patience, batch_size=batch_size,
        volumes_per_batch=volumes_per_batch,
        cycles_per_volume=cycles_per_volume, device=device,
        num_cpu_workers=num_cpu_workers, taskman_managed=taskman_managed,
        seed=seed)

    # Run the experiment
    experiment.load_dataset()
    experiment.build_model()
    experiment.train()
    experiment.save()


if __name__ == '__main__':
    main()
