#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import add_logging_arg
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.batch_loaders import prepare_batch_loader
from dwi_ml.training.utils.batch_samplers import prepare_batch_sampler
from dwi_ml.training.utils.experiment import add_args_resuming_experiment
from dwi_ml.training.utils.trainer import run_experiment

# Please adapt:
from dwi_ml.tests.utils import TrackingModelForTestWithPD


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
    model = TrackingModelForTestWithPD.load_params_and_state(
        os.path.join(args.experiments_path, args.experiment_name,
                     'checkpoint/model'),
        sub_loggers_level)

    # Prepare batch sampler
    _args = argparse.Namespace(**checkpoint_state['batch_sampler_params'])
    logging.warning(_args)
    batch_sampler = prepare_batch_sampler(dataset, _args, sub_loggers_level)

    # Prepare batch loader
    _args = argparse.Namespace(**checkpoint_state['batch_loader_params'])
    batch_loader = prepare_batch_loader(dataset, model, _args, sub_loggers_level)

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
