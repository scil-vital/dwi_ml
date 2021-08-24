#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
           Train a model for your favorite experiment

Remove or add parameters to fit your needs. You should change your yaml file
accordingly.

- Change init_dataset() if the MultiSubjectDataset doesn't fit your needs
- Change init_sampler() for the BatchSampler version that fits your needs
- Implement build_model()
- Change the DWIMLTrainer if it doesn't fit your needs.
"""

import argparse
import logging
import os
from os import path

import yaml

from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.experiment.timer import Timer
from dwi_ml.model.main_models import ModelAbstract
from dwi_ml.training.checks_for_experiment_parameters import (
    check_all_experiment_parameters, check_logging_level)
from dwi_ml.training.trainers import (DWIMLTrainer)

# These are model-dependant. Choose the best classes and functions for you
# 1. Change this init_dataset function if you prefer! This loads either a
#    MultiSubjectDataset or a LazyMultiSubjectDataset.
from dwi_ml.data.dataset.multi_subject_containers import init_dataset
# 2. Change this init_batch_sampler if you prefer!
#    This is one possibility, others could be implemented.
from dwi_ml.model.batch_samplers import (
    BatchSequencesSamplerOneInputVolume as ChosenBatchSampler)


# 3. Implement the build_model function below


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Path where to save your experiment. Complete path '
                        'will be experiment_path/experiment_name.')
    p.add_argument('--hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects. If a '
                        'checkpoint exists, this information is already '
                        'contained in the checkpoint and is not necessary.')
    p.add_argument('--parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for '
                        'an example. If a checkpoint exists, this information '
                        'is already contained in the checkpoint and is not '
                        'necessary.')
    p.add_argument('--experiment_name',
                   help='If given, name for the experiment. Else, model will '
                        'decide the name to give based on time of day.')
    p.add_argument('--override_checkpoint_patience', type=int,
                   help='If a checkpoint exists, patience can be increased '
                        'to allow experiment to continue if the allowed '
                        'number of bad epochs has been previously reached.')

    arguments = p.parse_args()

    return arguments


def build_model(input_size, **_):
    model = ModelAbstract()

    return model


def prepare_data_and_model(train_data_params, valid_data_params,
                           train_sampler_params, valid_sampler_params,
                           model_params):
    # Instantiate dataset classes
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        training_dataset = init_dataset(**train_data_params)
        validation_dataset = init_dataset(**valid_data_params)

    # Instantiate batch
    with Timer("\n\nPreparing batch samplers with volume: {}"
                       .format(training_dataset.volume_groups[0]),
               newline=True, color='green'):
        training_batch_sampler = ChosenBatchSampler(
            training_dataset, training_dataset.streamline_group,
            training_dataset.volume_groups[0], **train_sampler_params)
        validation_batch_sampler = ChosenBatchSampler(
            validation_dataset, validation_dataset.streamline_group,
            validation_dataset.volume_groups[0], **valid_sampler_params)

    # Instantiate model.
    # Hint: input_size is :
    size = training_batch_sampler.data_source.data_list.feature_sizes[0]
    # Instantiate batch
    with Timer("\n\nPreparing model",
               newline=True, color='yellow'):
        model = build_model(size)

    return training_batch_sampler, validation_batch_sampler, model


def main():
    args = parse_args()

    # Check that all files exist
    if not path.exists(args.hdf5_filename):
        raise FileNotFoundError("The hdf5 file ({}) was not found!"
                                .format(args.hdf5_filename))
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("The Yaml parameters file was not found: {}"
                                .format(args.parameters_filename))

    # Load parameters
    with open(args.parameters_filename) as f:
        yaml_parameters = yaml.safe_load(f.read())

    # Initialize logger
    logging_level = check_logging_level(yaml_parameters['logging']['level'])
    logging.basicConfig(level=logging_level)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        # Loading checkpoint
        print("Experiment checkpoint folder exists, resuming experiment!")
        checkpoint_state = \
            DWIMLTrainer.load_params_from_checkpoint(args.experiment_path)
        if args.parameters_filename:
            logging.warning('Resuming experiment from checkpoint. Yaml file '
                            'option was not necessary and will not be used!')
        if args.hdf5_filename:
            logging.warning('Resuming experiment from checkpoint. hdf5 file '
                            'option was not necessary and will not be used!')

        # Stop now if early stopping was triggered.
        DWIMLTrainer.check_early_stopping(checkpoint_state,
                                          args.override_checkpoint_patience)

        # Prepare the trainer from checkpoint_state
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(
            checkpoint_state['train_data_params'],
            checkpoint_state['valid_data_params'],
            checkpoint_state['train_sampler_params'],
            checkpoint_state['valid_sampler_params'],
            None)

        # Instantiate trainer
        with Timer("\n\nPreparing trainer",
                   newline=True, color='red'):
            trainer = DWIMLTrainer.init_from_checkpoint(
                training_batch_sampler, validation_batch_sampler, model,
                checkpoint_state)
    else:
        # Perform checks
        # We have decided to use yaml for a more rigorous way to store
        # parameters, compared, say, to bash. However, no argparser is used so
        # we need to make our own checks.
        all_params = check_all_experiment_parameters(yaml_parameters)

        # Modifying params to copy the checkpoint_state params.
        # Each class (dataset, batch sampler, etc) will receive a lot of
        # non-useful parameters, but that's ok.
        # Params in all_params (coming from the yaml file) must have the same
        # keys as when using a checkpoint.
        all_params['hdf5_filename'] = args.hdf5_filename
        all_params['experiment_path'] = args.experiment_path
        all_params['experiment_name'] = args.experiment_name

        train_params = all_params.copy()
        train_params['subjs_set'] = 'training_subjs'
        valid_params = all_params.copy()
        valid_params['subjs_set'] = 'validation_subjs'

        # Prepare the trainer from checkpoint_state
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(train_params, valid_params,
                                         all_params, all_params,
                                         None)

        # Instantiate trainer
        with Timer("\n\nPreparing trainer",
                   newline=True, color='red'):
            trainer = DWIMLTrainer(training_batch_sampler,
                                   validation_batch_sampler, model,
                                   **all_params)

    #####
    # Run (or continue) the experiment
    #####
    try:
        with Timer("\n\n****** Training!!! ********",
                   newline=True, color='magenta'):
            trainer.train_validate_and_save_loss()
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    print("Script terminated successfully. Saved experiment in folder : ")
    print(trainer.experiment_dir)


if __name__ == '__main__':
    main()
