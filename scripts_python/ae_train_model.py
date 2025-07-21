#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Autoencoders
"""
import argparse
import logging
import os

# comet_ml not used, but comet_ml requires to be imported before torch.
# See bug report here https://github.com/Lightning-AI/lightning/issues/5829
# Importing now to solve issues later.
import comet_ml  # noqa F401
import torch

from scilpy.io.utils import (assert_inputs_exist, assert_outputs_exist,
                             add_verbose_arg)

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_memory_args
from dwi_ml.models.projects.ae_models import ModelAE
from dwi_ml.training.projects.ae_trainer import TrainerWithBundleDPS
from dwi_ml.training.utils.batch_samplers import (add_args_batch_sampler,
                                                  prepare_batch_sampler)
from dwi_ml.training.utils.batch_loaders import (add_args_batch_loader)
from dwi_ml.training.utils.trainer import (add_training_args, run_experiment,
                                           format_lr)
from dwi_ml.training.batch_loaders import DWIMLStreamlinesBatchLoader
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_experiment_and_hdf5_path)


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_mandatory_args_experiment_and_hdf5_path(p)
    add_args_batch_sampler(p)
    add_args_batch_loader(p)
    add_training_args(p)
    p.add_argument('streamline_group_name',
                   help="Name of the group in hdf5")
    p.add_argument('--viz_latent_space', action='store_true', default=False,
                   help="If specified, enables latent space visualization.\n")
    p.add_argument('--viz_color_by', type=str, default=None,
                   help="Name of the group in hdf5 to color by."
                   "In the HDF5, the coloring group should be under"
                   "data_per_streamline/<color_by>")
    p.add_argument('--viz_max_bundle_size', type=int, default=1000,
                   help="Maximum number of streamlines per bundle to "
                   "visualize in latent space. Will perform a random\n"
                   "selection if the number of streamlines is higher than "
                   "this value."
                   )
    p.add_argument('--viz_bundles_mapping', type=str, default=None,
                   help="Path to a txt file mapping bundles to a new name.\n"
                   "Each line of that file should be: <bundle_name> <number>")
    add_memory_args(p, add_lazy_options=True, add_rng=True)
    add_verbose_arg(p)

    return p


def init_from_args(args, sub_loggers_level):
    torch.manual_seed(args.rng)  # Set torch seed

    viz_latent_space = args.viz_latent_space
    viz_color_by = args.viz_color_by

    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args, load_testing=False,
                                          log_level=sub_loggers_level)

    # Preparing the model
    # (Direction getter)
    # (Nb features)
    # Final model
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        # INPUTS: verifying args
        model = ModelAE(
            experiment_name=args.experiment_name,
            log_level=sub_loggers_level)

        logging.info("AEmodel final parameters:" +
                     format_dict_to_str(model.params_for_checkpoint))

        logging.info("Computed parameters:" +
                     format_dict_to_str(model.computed_params_for_display))

    # Preparing the batch samplers
    batch_sampler = prepare_batch_sampler(dataset, args, sub_loggers_level)
    # Preparing the batch loader.
    with Timer("\nPreparing batch loader...", newline=True, color='pink'):
        batch_loader = DWIMLStreamlinesBatchLoader(
            dataset=dataset, model=model,
            streamline_group_name=args.streamline_group_name,
            # OTHER
            rng=args.rng, log_level=sub_loggers_level)

        logging.info("Loader user-defined parameters: " +
                     format_dict_to_str(batch_loader.params_for_checkpoint))

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        lr = format_lr(args.learning_rate)
        trainer = TrainerWithBundleDPS(
            model=model, experiments_path=args.experiments_path,
            experiment_name=args.experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rates=lr, weight_decay=args.weight_decay,
            optimizer=args.optimizer, max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience, patience_delta=args.patience_delta,
            from_checkpoint=False, clip_grad=args.clip_grad,
            # MEMORY
            nb_cpu_processes=args.nbr_processes, use_gpu=args.use_gpu,
            log_level=sub_loggers_level,
            viz_latent_space=viz_latent_space, color_by=viz_color_by,
            bundles_mapping_file=args.viz_bundles_mapping,
            max_viz_subset_size=args.viz_max_bundle_size)
        logging.info("Trainer params : " +
                     format_dict_to_str(trainer.params_for_checkpoint))

    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting log level to INFO maximum for sub-loggers, else it becomes ugly,
    # but we will set trainer to user-defined level.
    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    # General logging (ex, scilpy: Warning)
    logging.getLogger().setLevel(level=logging.WARNING)

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiments_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if os.path.exists(os.path.join(args.experiments_path, args.experiment_name,
                                   "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script ae_resume_training_from_checkpoint.py.")

    trainer = init_from_args(args, sub_loggers_level)

    run_experiment(trainer)


if __name__ == '__main__':
    main()
