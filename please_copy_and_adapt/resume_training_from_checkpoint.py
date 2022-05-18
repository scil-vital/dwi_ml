#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os
from os import path

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.training.trainers import DWIMLAbstractTrainer
from dwi_ml.training.utils.batch_loaders import \
    prepare_batchloadersoneinput_train_valid
from dwi_ml.training.utils.batch_samplers import \
    prepare_batchsamplers_train_valid
from dwi_ml.training.utils.trainer import run_experiment


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiments_path', metavar='p',
                   help='Path from where to load your experiment, and where to'
                        'save new results.\nComplete path will be '
                        'experiments_path/experiment_name.')
    p.add_argument('experiment_name', metavar='n',
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

    p.add_argument('--logging', dest='logging_choice',
                   choices=['error', 'warning', 'info', 'as_much_as_possible',
                            'debug'],
                   help="Logging level. Error, warning, info are as usual.\n"
                        "The other options are two equivalents of 'debug' "
                        "level. \nWith 'as_much_as_possible', we print the "
                        "debug level only when the final result is still "
                        "readable (even during parallel training and during "
                        "tqdm loop). 'debug' print everything always, "
                        "even if ugly.")

    return p


def init_from_checkpoint(args):

    # Loading checkpoint
    checkpoint_state = DWIMLAbstractTrainer.load_params_from_checkpoint(
        args.saving_path, args.experiment_name)

    # Stop now if early stopping was triggered.
    DWIMLAbstractTrainer.check_stopping_cause(
        checkpoint_state, args.new_patience, args.new_max_epochs)

    # Prepare data
    args_data = argparse.Namespace(**checkpoint_state['train_data_params'])
    dataset = prepare_multisubjectdataset(args_data)
    # toDo Verify that valid dataset is the same.
    #  checkpoint_state['valid_data_params']

    # Load model from checkpoint directory
    model = MainModelAbstract.load(os.path.join(args.saving_path,
                                                args.experiment_name,
                                                'checkpoint/model'))

    # Prepare batch samplers
    args_ts = argparse.Namespace(**checkpoint_state['train_sampler_params'])
    args_vs = None if checkpoint_state['valid_sampler_params'] is None else \
        argparse.Namespace(**checkpoint_state['valid_sampler_params'])
    training_batch_sampler, validation_batch_sampler = \
        prepare_batchsamplers_train_valid(dataset, args_ts, args_vs)

    # Prepare batch loaders
    args_tl = argparse.Namespace(**checkpoint_state['train_loader_params'])
    args_vl = None if checkpoint_state['valid_loader_params'] is None else \
        argparse.Namespace(**checkpoint_state['valid_loader_params'])
    training_batch_loader, validation_batch_loader = \
        prepare_batchloadersoneinput_train_valid(dataset, args_tl, args_vl)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLAbstractTrainer.init_from_checkpoint(
            model, args.saving_path, args.experiment_name,
            training_batch_sampler, training_batch_loader,
            validation_batch_sampler, validation_batch_loader,
            checkpoint_state, args.new_patience, args.new_max_epochs)
    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Initialize logger for preparation (loading data, model, experiment)
    # If 'as_much_as_possible', we will modify the logging level when starting
    # the training, else very ugly
    logging_level = args.logging_choice.upper()
    if args.logging_choice == 'as_much_as_possible':
        logging_level = 'DEBUG'
    logging.basicConfig(level=logging_level)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if not path.exists(os.path.join(args.experiments_path,
                                    args.experiment_name, "checkpoint")):
        raise FileNotFoundError("Experiment not found.")

    trainer = init_from_checkpoint(args)

    run_experiment(trainer, args.logging_choice)


if __name__ == '__main__':
    main()
