#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

# comet_ml not used, but comet_ml requires to be imported before torch.
# See bug report here https://github.com/Lightning-AI/lightning/issues/5829
# Importing now to solve issues later.
import comet_ml

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.projects.transforming_tractography import OriginalTransformerModel
from dwi_ml.training.projects.transformer_trainer import TransformerTrainer
from dwi_ml.training.utils.batch_samplers import prepare_batch_sampler
from dwi_ml.training.utils.batch_loaders import prepare_batch_loader
from dwi_ml.training.utils.trainer import run_experiment


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiments_path',
                   help='Path from where to load your experiment, and where to'
                        'save new results.\nComplete path will be '
                        'experiments_path/experiment_name.')
    p.add_argument('experiment_name',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to \ngive based on time of day.')

    p.add_argument('--new_patience', type=int, metavar='new_p',
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of bad epochs has been previously reached.')
    p.add_argument('--new_max_epochs', type=int,
                   metavar='new_max',
                   help='If a checkpoint exists, max_epochs can be increased '
                        'to allow experiment \nto continue if the allowed '
                        'number of epochs has been previously reached.')

    p.add_argument('--logging', dest='logging_choice', default='WARNING',
                   choices=['ERROR', 'WARNING', 'INFO', 'DEBUG'],
                   help="Logging level. Note that, for readability, not all "
                        "debug logs are printed in DEBUG mode.")

    return p


def init_from_checkpoint(args, checkpoint_path):

    # Loading checkpoint
    checkpoint_state = TransformerTrainer.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)

    # Stop now if early stopping was triggered.
    TransformerTrainer.check_stopping_cause(
        checkpoint_state, args.new_patience, args.new_max_epochs)

    # Prepare data
    args_data = argparse.Namespace(**checkpoint_state['dataset_params'])
    dataset = prepare_multisubjectdataset(args_data)

    # Setting log level to INFO maximum for sub-loggers, else it become ugly
    sub_loggers_level = args.logging_choice
    if args.logging_choice == 'DEBUG':
        sub_loggers_level = 'INFO'

    # Prepare model
    model = OriginalTransformerModel.load_params_and_state(
        os.path.join(checkpoint_path, 'model'), sub_loggers_level)

    # Prepare batch sampler
    _args = argparse.Namespace(**checkpoint_state['batch_sampler_params'])
    batch_sampler = prepare_batch_sampler(dataset, _args, sub_loggers_level)

    # Prepare batch loader
    _args = argparse.Namespace(**checkpoint_state['batch_loader_params'])
    batch_loader = prepare_batch_loader(dataset, model, _args, sub_loggers_level)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = TransformerTrainer.init_from_checkpoint(
            model, args.experiments_path, args.experiment_name,
            batch_sampler,  batch_loader,
            checkpoint_state, args.new_patience, args.new_max_epochs,
            args.logging_choice)
    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting root logger with high level, but we will set trainer to
    # user-defined level.
    logging.getLogger().setLevel(level=logging.INFO)

    # Verify if a checkpoint has been saved.
    checkpoint_path = os.path.join(
            args.experiments_path, args.experiment_name, "checkpoint")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Experiment's checkpoint not found ({})."
                                .format(checkpoint_path))

    trainer = init_from_checkpoint(args, checkpoint_path)

    run_experiment(trainer)


if __name__ == '__main__':
    main()
