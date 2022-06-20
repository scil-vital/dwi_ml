#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import add_logging_arg
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.batch_loaders import \
    prepare_batchloadersoneinput
from dwi_ml.training.utils.batch_samplers import \
    prepare_batchsampler
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
    args_data = argparse.Namespace(**checkpoint_state['train_data_params'])
    dataset = prepare_multisubjectdataset(args_data)
    # toDo Verify that valid dataset is the same.
    #  checkpoint_state['valid_data_params']

    # Setting log level to INFO maximum for sub-loggers, else it become ugly
    sub_loggers_level = args.logging
    if args.logging == 'DEBUG':
        sub_loggers_level = 'INFO'

    # Load model from checkpoint directory
    model = ModelForTestWithPD.load(
        os.path.join(args.experiments_path, args.experiment_name,
                     'checkpoint/model'),
        sub_loggers_level)

    # Prepare batch samplers
    args_ts = checkpoint_state['train_sampler_params']
    args_vs = None if checkpoint_state['valid_sampler_params'] is None else \
        checkpoint_state['valid_sampler_params']
    with Timer("\nPreparing batch samplers...", newline=True, color='green'):
        logging.info("Instantiating training set's batch sampler...")

        batch_sampler = DWIMLBatchSampler(
            dataset, streamline_group_name=args['streamline_group_name'],
            batch_size_training=args['batch_size_training'],
            batch_size_validation=args['batch_size_validation'],
            batch_size_units=args['batch_size_units'],
            nb_streamlines_per_chunk=args['nb_streamlines_per_chunk'],
            nb_subjects_per_batch=args['nb_subjects_per_batch'],
            cycles=args['cycles'],
            rng=args['rng'], log_level=sub_loggers_level)

        logging.info("Batch sampler's user-defined parameters: " +
                     format_dict_to_str(batch_sampler.params))

    # Prepare batch loaders
    args_tl = checkpoint_state['train_loader_params']
    args_vl = None if checkpoint_state['valid_loader_params'] is None else \
        checkpoint_state['valid_loader_params']
    with Timer("\nPreparing batch loaders...", newline=True, color='pink'):
        logging.info("Instantiating training set's batch loader...")

        batch_loader = BatchLoaderOneInput(
            dataset, input_group_name=args['input_group_name'],
            streamline_group_name=args['streamline_group_name'],
            # STREAMLINES PREPROCESSING
            step_size=args['step_size'], compress=args['compress'],
            # STREAMLINES AUGMENTATION
            noise_gaussian_size_training=args['noise_gaussian_size_training'],
            noise_gaussian_variability_training=args['noise_gaussian_variability_training'],
            noise_gaussian_size_validation=args['noise_gaussian_size_validation'],
            noise_gaussian_variability_validation=args['noise_gaussian_variability_validation'],
            reverse_ratio=args['reverse_ratio'], split_ratio=args['split_ratio'],
            # NEIGHBORHOOD
            neighborhood_points=args['neighborhood_points'],
            # OTHER
            rng=args['rng'], wait_for_gpu=args['wait_for_gpu'],
            log_level=log_level)

        logging.info("Loader user-defined parameters: " +
                     format_dict_to_str(batch_loader.params_for_json_prints))

    # Instantiate trainer
    with Timer("\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLTrainerOneInput.init_from_checkpoint(
            model, args.experiments_path, args.experiment_name,
            training_batch_sampler, training_batch_loader,
            validation_batch_sampler, validation_batch_loader,
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

    run_experiment(trainer, args.logging)


if __name__ == '__main__':
    main()
