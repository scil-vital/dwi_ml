#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# Remove or add parameters to fit your needs. You should change your yaml file
# accordingly.
# Change StreamlinesBasedModelAbstract for an implementation of your own model.
# It should be a child of this abstract class.
############################

""" Train a model for my favorite experiment"""

import argparse
import logging
from os import path
import yaml

from dwi_ml.training.scripts_utils import *
from dwi_ml.training.trainer_abstract import StreamlinesBasedModelAbstract


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for an '
                        'example.')

    arguments = p.parse_args()
    return arguments


def check_all_parameters(conf):

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
    add_x_previous_dirs = check_previous_dir(
        conf['input']['add_x_previous_dirs'])

    # Epochs:
    max_epochs = check_max_epochs(
        conf['epochs']['max_epochs'], required=False)
    patience = check_patience(
        conf['epochs']['patience'], required=False)
    batch_size = check_batch_size(
        conf['epochs']['batch']['size'], required=False)
    volumes_per_batch = check_volumes_per_batch(
        conf['epochs']['batch']['volumes_used'], required=False)
    cycles_per_volume = check_cycles_per_volume(
        conf['epochs']['batch']['cycles_per_volume'], volumes_per_batch)

    # Memory:
    lazy = check_lazy(
        conf['memory']['lazy'])
    cache_manager = check_cache_manager(
        conf['memory']['cache_manager'], lazy)
    use_gpu = check_use_gpu(
        conf['memory']['use_gpu'])
    num_cpu_workers = check_num_cpu_workers(
        conf['memory']['num_workers'], required=False)
    worker_interpolation = check_worker_interpolation(
        conf['memory']['worker_interpolation'])
    taskman_managed = check_taskman_managed(
        conf['memory']['taskman_managed'], required=False)

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
        'add_x_previous_dirs': add_x_previous_dirs,
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


def main():
    args = parse_args()

    # Load parameters from yaml file
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("Yaml file not found: "
                                "{}".format(args.parameters_filename))
    with open(args.experiment_conf_fname) as f:
        conf = yaml.safe_load(f.read())

    # Initialize logger
    logging_level = check_logging_level(conf['logging']['level'],
                                        required=False)
    logging.basicConfig(level=logging_level)
    logging.info(conf)

    # Perform checks
    organized_args = check_all_parameters(conf)

    # Instantiate your class
    # (Change StreamlinesBasedModelAbstract for your class.)
    # Then load dataset, build model, train and save
    experiment = StreamlinesBasedModelAbstract(organized_args)
    experiment.load_dataset()
    experiment.build_model()
    experiment.train()
    experiment.save()


if __name__ == '__main__':
    main()
