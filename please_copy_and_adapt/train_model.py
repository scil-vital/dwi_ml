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

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment.monitoring import EarlyStoppingError
from dwi_ml.experiment.timer import Timer
from dwi_ml.model.main_models import MainModelAbstract
from dwi_ml.experiment.checks_for_experiment_parameters import (
    check_all_experiment_parameters)
from dwi_ml.training.trainers import (DWIMLTrainer)
from dwi_ml.utils import format_dict_to_str

# These are model-dependant. Choose the best classes and functions for you
# 1. Change this init_batch_sampler if you prefer!
#    This is one possibility, others could be implemented.
from dwi_ml.model.batch_samplers import (
    BatchStreamlinesSampler1IPV as ChosenBatchSampler)
# 2. Implement the build_model function below


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Path where to save your experiment. Complete path '
                        'will be experiment_path/experiment_name.')
    p.add_argument('--input_group',
                   help='Name of the input group. If a checkpoint exists, '
                        'this information is already contained in the '
                        'checkpoint and is not necessary.')
    p.add_argument('--target_group',
                   help='Name of the target streamline group. If a checkpoint '
                        'exists, this information is already contained in the '
                        'checkpoint and is not necessary.')
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
    p.add_argument('--logging', choices=['error', 'warning', 'info', 'debug'],
                   help="Logging level. One of ['error', 'warning', 'info', "
                        "'debug']. Default: Info.")
    p.add_argument('--comet_workspace',
                   help='Your comet workspace. If not set, comet.ml will not '
                        'be used. See our docs/Getting Started for more '
                        'information on comet and its API key.')
    p.add_argument('--comet_project',
                   help='Send your experiment to a specific comet.ml project. '
                        'If not set, it will be sent to Uncategorized '
                        'Experiments.')

    arguments = p.parse_args()

    return arguments


def build_model(input_size, **_):
    model = MainModelAbstract()

    return model


def prepare_data_and_model(dataset_params, train_sampler_params,
                           valid_sampler_params, model_params):
    # Instantiate dataset classes
    with Timer("\n\nPreparing testing and validation sets",
               newline=True, color='blue'):
        dataset = MultiSubjectDataset(**dataset_params)
        dataset.load_data()

        logging.info("\n\nDataset attributes: \n" +
                     format_dict_to_str(dataset.attributes))

    # Instantiate batch
    # In this example, using one input volume, the first one.
    volume_group_name = train_sampler_params['input_group_name']
    volume_group_idx = dataset.volume_groups.index(volume_group_name)
    with Timer("\n\nPreparing batch samplers with volume: '{}' and "
               "streamlines '{}'"
               .format(volume_group_name,
                       train_sampler_params['streamline_group_name']),
               newline=True, color='green'):
        training_batch_sampler = ChosenBatchSampler(dataset.training_set,
                                                    **train_sampler_params)
        validation_batch_sampler = ChosenBatchSampler(dataset.validation_set,
                                                      **valid_sampler_params)
        logging.info("\n\nTraining batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))
        logging.info("\n\nValidation batch sampler attributes: \n" +
                     format_dict_to_str(training_batch_sampler.attributes))

    # Instantiate model.
    input_size = dataset.nb_features[volume_group_idx]
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = build_model(input_size, **model_params)

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

    # Initialize logger
    if args.logging:
        level = args.logging.upper()
    else:
        level = 'INFO'
    logging.basicConfig(level=level)

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
            checkpoint_state['dataset_params'],
            checkpoint_state['train_sampler_params'],
            checkpoint_state['valid_sampler_params'],
            None)

        # Instantiate trainer
        with Timer("\n\nPreparing trainer", newline=True, color='red'):
            trainer = DWIMLTrainer.init_from_checkpoint(
                training_batch_sampler, validation_batch_sampler, model,
                checkpoint_state)
    else:
        # Load parameters
        with open(args.parameters_filename) as f:
            yaml_parameters = yaml.safe_load(f.read())

        # Perform checks
        # We have decided to use yaml for a more rigorous way to store
        # parameters, compared, say, to bash. However, no argparser is used so
        # we need to make our own checks.
        (sampling_params, training_params, model_params, memory_params,
         randomization) = check_all_experiment_parameters(yaml_parameters)

        # Modifying params to copy the checkpoint_state params.
        # Params coming from the yaml file must have the same keys as when
        # using a checkpoint.
        experiment_params = {
            'hdf5_filename': args.hdf5_filename,
            'experiment_path': args.experiment_path,
            'experiment_name': args.experiment_name,
            'comet_workspace': args.comet_workspace,
            'comet_project': args.comet_project}

        # MultiSubjectDataset parameters
        dataset_params = {**memory_params,
                          **experiment_params}

        # BatchSampler parameters
        # If you wish to have different parameters for the batch sampler during
        # trainnig and validation, change values below.
        sampler_params = {**sampling_params,
                          **model_params,
                          **randomization,
                          **memory_params,
                          'input_group_name': args.input_group,
                          'streamline_group_name': args.target_group,
                          'wait_for_gpu': memory_params['use_gpu']}

        model_params.update(memory_params)

        # Prepare the trainer from params
        (training_batch_sampler, validation_batch_sampler,
         model) = prepare_data_and_model(dataset_params, sampler_params,
                                         sampler_params, model_params)

        # Instantiate trainer
        with Timer("\n\nPreparing trainer", newline=True, color='red'):
            trainer = DWIMLTrainer(
                training_batch_sampler, validation_batch_sampler, model,
                **training_params, **experiment_params, **memory_params,
                from_checkpoint=False)

    #####
    # Run (or continue) the experiment
    #####
    try:
        with Timer("\n\n****** Running model!!! ********",
                   newline=True, color='magenta'):
            trainer.run_model()
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    print("Script terminated successfully. Saved experiment in folder : ")
    print(trainer.experiment_dir)


if __name__ == '__main__':
    main()
