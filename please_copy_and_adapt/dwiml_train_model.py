#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model.

This script uses a fake model to allow testing. Please create your own model
(possibly with your own trainer, batch sampler, batch loader), copy this script
to your project and adapt it.
"""
import argparse
import logging
import os

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import (
    add_dataset_args, prepare_multisubjectdataset)
from dwi_ml.experiment_utils.prints import add_logging_arg, format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.batch_samplers import (
    add_args_batch_sampler, prepare_batchsamplers_train_valid)
from dwi_ml.training.utils.batch_loaders import (
    add_args_batch_loader, prepare_batchloadersoneinput_train_valid)
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_training_experiment,
    add_memory_args_training_experiment)
from dwi_ml.training.utils.trainer import add_training_args, run_experiment

# Please adapt
from dwi_ml.tests.utils import ModelForTestWithPD


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_mandatory_args_training_experiment(p)
    add_logging_arg(p)
    add_memory_args_training_experiment(p)
    add_dataset_args(p)
    add_args_batch_sampler(p)
    add_args_batch_loader(p)
    add_training_args(p)

    # To be defined: add_model_args(p)
    # Possibly:
    # models.utils.direction_getters.add_direction_getter_args(p)
    # data.processing.space.neighborhood.add_args_neighborhood(p)

    return p


def init_from_args(args, sub_loggers_level):


    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args, load_testing=False,
                                          log_level=sub_loggers_level)

    # Preparing the model
    # Possibly useful:
    #     input_group_idx = dataset.volume_groups.index(args.input_group_name)
    #     nb_features = dataset.nb_features[input_group_idx]
    #     dg_args = check_args_direction_getter(args)
    model = ModelForTestWithPD()  # To be instantiated correctly.

    # Prepare args.
    logging.debug("Initial {}".format(args))

    # Preparing the batch samplers.
    # The only parameter that may differ is the batch size.
    args_sampler = {
        'batch_size_units': args.batch_size_units,
        'streamline_group_name': args.streamline_group_name,
        'nb_streamlines_per_chunk': args.nb_streamlines_per_chunk,
        'nb_subjects_per_batch': args.nb_subjects_per_batch,
        'cycles': args.cycles,
        'rng': args.rng
    }

    args_sampler_training = {
        **args_sampler,
        'batch_size': args.batch_size_training}
    args_sampler_validation = {
        **args_sampler,
        'batch_size': args.batch_size_training}

    training_batch_sampler, validation_batch_sampler = \
        prepare_batchsamplers_train_valid(dataset,
                                          args_sampler_training,
                                          args_sampler_validation,
                                          sub_loggers_level)

    # Preparing the batch loaders
    # The only parameter that may differ is the gaussian noise.
    args_loader = {
        'input_group_name': args.input_group_name,
        'streamline_group_name': args.streamline_group_name,
        'step_size': args.step_size,
        'compress': args.compress,
        'reverse_ratio': args.reverse_ratio,
        'split_ratio': args.split_ratio,
        'neighborhood_points': model.neighborhood_points,
        'rng': args.rng,
        'wait_for_gpu': args.use_gpu
    }
    args_loader_train = {
        **args_loader,
        'noise_gaussian_size': args.noise_gaussian_size_training,
        'noise_gaussian_variability': args.noise_gaussian_variability_training}
    args_loader_valid = {
        **args_loader,
        'noise_gaussian_size': args.noise_gaussian_size_validation,
        'noise_gaussian_variability': args.noise_gaussian_variability_validation}

    training_batch_loader, validation_batch_loader = \
        prepare_batchloadersoneinput_train_valid(dataset, args_loader_train,
                                                 args_loader_valid,
                                                 sub_loggers_level)

    # Instantiate trainer
    model_uses_streamlines = True  # Our test model uses previous dirs, so
    # streamlines need to be sent to the forward method.
    with Timer("\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLTrainerOneInput(
            model, args.experiments_path, args.experiment_name,
            training_batch_sampler, training_batch_loader,
            validation_batch_sampler, validation_batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            model_uses_streamlines=model_uses_streamlines,
            learning_rate=args.learning_rate, max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience, from_checkpoint=False,
            weight_decay=args.weight_decay,
            # MEMORY
            nb_cpu_processes=args.processes, use_gpu=args.use_gpu,
            log_level=args.logging)
        logging.info("Trainer params : " +
                     format_dict_to_str(trainer.params_for_json_prints))

    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting log level to INFO maximum for sub-loggers, else it become ugly
    # but we will set trainer to user-defined level.
    sub_loggers_level = args.logging
    if args.logging == 'DEBUG':
        sub_loggers_level = 'INFO'

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiments_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if os.path.exists(os.path.join(args.experiments_path, args.experiment_name,
                                   "checkpoint")):
        raise FileExistsError(
            "This experiment already exists. Delete or use script "
            "dwiml_resume_training_from_checkpoint.py.")

    trainer = init_from_args(args, sub_loggers_level)

    run_experiment(trainer, args.logging)


if __name__ == '__main__':
    main()
