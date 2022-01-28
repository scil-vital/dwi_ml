#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model.

See an example of the yaml file in the parameters folder.
"""
import argparse
import logging
import os
from os import path

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import (
    add_args_dataset, prepare_multisubjectdataset)
from dwi_ml.training.utils.batch_samplers import (
    add_args_batch_sampler, prepare_batchsamplers_train_valid)
from dwi_ml.training.utils.batch_loaders import (
    add_args_batch_loader, prepare_batchloadersoneinput_train_valid)
from dwi_ml.training.utils.trainer import (add_training_args, prepare_trainer,
                                           run_experiment)


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument(
        'experiment_path',
        help='Path where to save your experiment. \nComplete path will be '
             'experiment_path/experiment_name.')
    p.add_argument(
        'experiment_name',
        help='If given, name for the experiment. Else, model will decide the '
             'name to \ngive based on time of day.')
    p.add_argument(
        'hdf5_file',
        help='Path to the .hdf5 dataset. Should contain both your training '
             'and \nvalidation subjects.')
    p.add_argument(
        'input_group_name',
        help='Name of the input volume in the hdf5 dataset.')
    p.add_argument(
        'streamline_group_name',
        help="Name of the streamlines group in the hdf5 dataset.")

    p.add_argument(
        '--logging', dest='logging_choice',
        choices=['error', 'warning', 'info', 'as_much_as_possible', 'debug'],
        help="Logging level. Error, warning, info are as usual.\n The other "
             "options are two equivalents of 'debug' level. \nWith "
             "'as_much_as_possible', we print the debug level only when the "
             "final \nresult is still readable (even during parallel training "
             "and during tqdm \nloop). 'debug' prints everything always, even "
             "if ugly.")
    p.add_argument(
        '--taskman_managed', action='store_true',
        help="If set, instead of printing progression, print taskman-relevant "
             "data.")

    # Memory options both for the batch sampler and the trainer:
    m_g = p.add_argument_group("Memory options :")
    m_g.add_argument(
        '--use_gpu', action='store_true',
        help="If set, avoids computing and interpolating the inputs (and "
             "their neighborhood) \ndirectly in the batch sampler (which is "
             "computed on CPU). Will be computed later on GPU, \nin the "
             "trainer.")
    m_g.add_argument(
        '--processes', type=int, default=1,
        help="Number of parallel CPU processes, when working on CPU.")
    m_g.add_argument(
        '--rng', type=int, default=1234,
        help="Random seed. Default=1234.")

    add_args_dataset(p)
    add_model_args(p)
    add_args_batch_sampler(p)
    add_args_batch_loader(p)
    add_training_args(p)

    return p


def init_from_args(p, args):
    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args)

    # Preparing the model
    if args.grid_radius:
        args.neighborhood_radius = args.grid_radius
        args.neighborhood_type = 'grid'
    elif args.sphere_radius:
        args.neighborhood_radius = args.sphere_radius
        args.neighborhood_type = 'axes'
    else:
        args.neighborhood_radius = None
        args.neighborhood_type = None
    input_group_idx = dataset.volume_groups.index(args.input_group_name)
    args.nb_features = dataset.nb_features[input_group_idx]
    model = prepare_model(args)

    # Preparing the batch samplers
    args.wait_for_gpu = args.use_gpu
    training_batch_sampler, validation_batch_sampler = \
        prepare_batchsamplers_train_valid(dataset, args, args)

    # Preparing the batch loaders
    args.neighborhood_points = model.neighborhood_points
    training_batch_loader, validation_batch_loader = \
        prepare_batchloadersoneinput_train_valid(dataset, args, args)

    # Preparing the trainer
    trainer = prepare_trainer(training_batch_sampler, validation_batch_sampler,
                              training_batch_loader, validation_batch_loader,
                              model, args)

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

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiment_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if path.exists(os.path.join(args.experiment_path, args.experiment_name,
                                "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script resume_training_from_checkpoint.py.")

    trainer = init_from_args(p, args)

    run_experiment(trainer, args.logging_choice)


if __name__ == '__main__':
    main()
