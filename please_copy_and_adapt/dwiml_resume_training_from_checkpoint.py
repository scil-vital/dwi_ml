#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import add_logging_arg, format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.experiment import add_args_resuming_experiment
from dwi_ml.training.utils.trainer import run_experiment

# Please adapt:
from dwi_ml.tests.utils import ModelForTestWithPD


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_args_resuming_experiment(p)

    add_logging_arg(p)

    return p


def init_from_checkpoint(args):

    # Loading checkpoint
    checkpoint_state = DWIMLTrainerOneInput.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)

    # Stop now if early stopping was triggered.
    DWIMLTrainerOneInput.check_stopping_cause(
        checkpoint_state, args.new_patience, args.new_max_epochs)

    # Prepare data
    args_data = argparse.Namespace(**checkpoint_state['dataset_params'])
    dataset = prepare_multisubjectdataset(args_data)

    # Setting log level to INFO maximum for sub-loggers, else it become ugly
    sub_loggers_level = args.logging
    if args.logging == 'DEBUG':
        sub_loggers_level = 'INFO'

    # Load model from checkpoint directory
    model = ModelForTestWithPD.load_params_and_state(
        os.path.join(args.experiments_path, args.experiment_name,
                     'checkpoint/model'),
        sub_loggers_level)

    # Prepare batch sampler
    _args = checkpoint_state['batch_sampler_params']
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Instantiating training set's batch sampler...")

        batch_sampler = DWIMLBatchIDSampler(
            dataset, streamline_group_name=_args['streamline_group_name'],
            batch_size_training=_args['batch_size_training'],
            batch_size_validation=_args['batch_size_validation'],
            batch_size_units=_args['batch_size_units'],
            nb_streamlines_per_chunk=_args['nb_streamlines_per_chunk'],
            nb_subjects_per_batch=_args['nb_subjects_per_batch'],
            cycles=_args['cycles'],
            rng=_args['rng'], log_level=sub_loggers_level)

        logging.info("Batch sampler's user-defined parameters: " +
                     format_dict_to_str(batch_sampler.params))

    # Prepare batch loader
    _args = checkpoint_state['batch_loader_params']
    with Timer("\nPreparing batch loaders...", newline=True, color='pink'):
        logging.info("Instantiating training set's batch loader...")

        batch_loader = DWIMLBatchLoaderOneInput(
            dataset, input_group_name=_args['input_group_name'],
            streamline_group_name=_args['streamline_group_name'],
            # STREAMLINES PREPROCESSING
            step_size=_args['step_size'], compress=_args['compress'],
            # STREAMLINES AUGMENTATION
            noise_gaussian_size_training=_args['noise_gaussian_size_training'],
            noise_gaussian_var_training=_args['noise_gaussian_var_training'],
            noise_gaussian_size_validation=_args['noise_gaussian_size_validation'],
            noise_gaussian_var_validation=_args['noise_gaussian_var_validation'],
            reverse_ratio=_args['reverse_ratio'],
            split_ratio=_args['split_ratio'],
            # NEIGHBORHOOD
            neighborhood_points=_args['neighborhood_points'],
            # OTHER
            rng=_args['rng'], wait_for_gpu=_args['wait_for_gpu'],
            log_level=sub_loggers_level)

        logging.info("Loader user-defined parameters: " +
                     format_dict_to_str(batch_loader.params_for_json_prints))

    # Instantiate trainer
    with Timer("\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLTrainerOneInput.init_from_checkpoint(
            model, args.experiments_path, args.experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, args.new_patience, args.new_max_epochs,
            args.logging)

    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting root logger with high level but we will set trainer to
    # user-defined level.
    logging.basicConfig(level=logging.WARNING)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if not os.path.exists(os.path.join(
            args.experiments_path, args.experiment_name, "checkpoint")):
        raise FileNotFoundError("Experiment not found.")

    trainer = init_from_checkpoint(args)

    run_experiment(trainer)


if __name__ == '__main__':
    main()
