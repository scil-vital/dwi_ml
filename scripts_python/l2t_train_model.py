#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a model for Learn2Track
"""
import argparse
import logging
import os

# comet_ml not used, but comet_ml requires to be imported before torch.
# See bug report here https://github.com/Lightning-AI/lightning/issues/5829
# Importing now to solve issues later.
import comet_ml
import torch
from dwi_ml.models.projects.learn2track_utils import \
    get_all_args_groups_learn2track

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.arg_utils import add_args_groups_to_parser
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.utils.direction_getters import check_args_direction_getter
from dwi_ml.training.projects.learn2track_trainer import Learn2TrackTrainer
from dwi_ml.training.utils.batch_samplers import prepare_batch_sampler
from dwi_ml.training.utils.batch_loaders import prepare_batch_loader
from dwi_ml.training.utils.trainer import run_experiment, format_lr, \
    get_training_args


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)

    groups = get_all_args_groups_learn2track()
    add_args_groups_to_parser(groups, p)

    return p


def init_from_args(args, sub_loggers_level):
    torch.manual_seed(args.rng)  # Set torch seed

    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args, load_testing=False,
                                          log_level=sub_loggers_level)

    # Preparing the model
    # (Direction getter)
    dg_args = check_args_direction_getter(args)
    # (Nb features)
    input_group_idx = dataset.volume_groups.index(args.input_group_name)
    args.nb_features = dataset.nb_features[input_group_idx]
    # Final model
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        # INPUTS: verifying args
        model = Learn2TrackModel(
            experiment_name=args.experiment_name,
            step_size=args.step_size, compress_lines=args.compress,
            # PREVIOUS DIRS
            prev_dirs_embedding_key=args.prev_dirs_embedding_key,
            prev_dirs_embedded_size=args.prev_dirs_embedded_size,
            nb_previous_dirs=args.nb_previous_dirs,
            normalize_prev_dirs=args.normalize_prev_dirs,
            # INPUTS
            input_embedding_key=args.input_embedding_key,
            input_embedded_size=args.input_embedded_size,
            nb_features=args.nb_features, kernel_size=args.kernel_size,
            nb_cnn_filters=args.nb_cnn_filters,
            # RNN
            rnn_key=args.rnn_key, rnn_layer_sizes=args.rnn_layer_sizes,
            dropout=args.dropout,
            use_layer_normalization=args.use_layer_normalization,
            use_skip_connection=args.use_skip_connection,
            start_from_copy_prev=args.start_from_copy_prev,
            # TRACKING MODEL
            dg_key=args.dg_key, dg_args=dg_args,
            # Other
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            neighborhood_resolution=args.neighborhood_resolution,
            log_level=sub_loggers_level)

        logging.info("Learn2track model final parameters:" +
                     format_dict_to_str(model.params_for_checkpoint))

        logging.info("Computed parameters:" +
                     format_dict_to_str(model.computed_params_for_display))

    # Preparing the batch samplers
    batch_sampler = prepare_batch_sampler(dataset, args, sub_loggers_level)
    batch_loader = prepare_batch_loader(dataset, model, args, sub_loggers_level)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        lr = format_lr(args.learning_rate)
        trainer = Learn2TrackTrainer(
            model=model, experiments_path=args.experiments_path,
            experiment_name=args.experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rates=lr, lr_decrease_params=args.lr_decrease_params,
            weight_decay=args.weight_decay,
            optimizer=args.optimizer, max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience, patience_delta=args.patience_delta,
            from_checkpoint=False, clip_grad=args.clip_grad,
            # (generation validation:)
            add_a_tracking_validation_phase=args.add_a_tracking_validation_phase,
            tracking_phase_frequency=args.tracking_phase_frequency,
            tracking_phase_nb_segments_init=args.tracking_phase_nb_segments_init,
            tracking_phase_mask_group=args.tracking_mask,
            # MEMORY
            nb_cpu_processes=args.nbr_processes, use_gpu=args.use_gpu,
            log_level=args.logging)
        logging.info("Trainer params : " +
                     format_dict_to_str(trainer.params_for_checkpoint))

    return trainer


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Setting log level to INFO maximum for sub-loggers, else it becomes ugly,
    # but we will set trainer to user-defined level.
    sub_loggers_level = args.logging
    if args.logging == 'DEBUG':
        sub_loggers_level = 'INFO'

    logging.getLogger().setLevel(level=logging.INFO)

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiments_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if os.path.exists(os.path.join(args.experiments_path, args.experiment_name,
                                   "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script l2t_resume_training_from_checkpoint.py.")

    trainer = init_from_args(args, sub_loggers_level)

    run_experiment(trainer)


if __name__ == '__main__':
    main()
