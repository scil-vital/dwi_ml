#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model.

This script uses a fake model to allow testing. Please create your own model
(possibly with your own trainer, batch sampler, batch loader), copy this script
to your projects and adapt it.
"""
import argparse
import logging
import os

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import (add_dataset_args,
                                       prepare_multisubjectdataset)
from dwi_ml.experiment_utils.prints import add_logging_arg, format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.batch_samplers import (add_args_batch_sampler,
                                                  prepare_batch_sampler)
from dwi_ml.training.utils.batch_loaders import (add_args_batch_loader,
                                                 prepare_batch_loader)
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_training_experiment,
    add_memory_args_training_experiment)
from dwi_ml.training.utils.trainer import add_training_args, run_experiment

# Please adapt
from dwi_ml.tests.utils.data_and_models_for_tests import TrackingModelForTestWithPD


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

    TrackingModelForTestWithPD.add_args_tracking_model(p)
    TrackingModelForTestWithPD.add_args_model_with_pd(p)
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
    model = TrackingModelForTestWithPD()  # To be instantiated correctly.

    # Preparing the batch sampler.
    batch_sampler = prepare_batch_sampler(dataset, args, sub_loggers_level)
    batch_loader = prepare_batch_loader(dataset, model, args, sub_loggers_level)

    # Instantiate trainer
    # streamlines need to be sent to the forward method.
    with Timer("\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLTrainerOneInput(
            model=model, experiments_path=args.experiments_path,
            experiment_name=args.experiment_name,
            batch_sampler=batch_sampler, batch_loader=batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rate=args.learning_rate, weight_decay=args.weight_decay,
            use_radam=args.use_radam, max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience, from_checkpoint=False,
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

    run_experiment(trainer)


if __name__ == '__main__':
    main()
