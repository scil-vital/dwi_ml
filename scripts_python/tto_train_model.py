#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a Transformer (original) model.
"""
import argparse
import logging
import os

# comet_ml not used, but comet_ml requires to be imported before torch.
# See bug report here https://github.com/Lightning-AI/lightning/issues/5829
# Importing now to solve issues later.
import comet_ml
import torch

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.io_utils import add_memory_args, add_logging_arg
from dwi_ml.models.projects.transforming_tractography import \
    OriginalTransformerModel
from dwi_ml.models.projects.transformers_utils import (add_abstract_model_args,
                                                       add_tto_model_args,
                                                       perform_checks)
from dwi_ml.training.projects.transformer_trainer import TransformerTrainer
from dwi_ml.training.utils.batch_samplers import (add_args_batch_sampler,
                                                  prepare_batch_sampler)
from dwi_ml.training.utils.batch_loaders import (add_args_batch_loader,
                                                 prepare_batch_loader)
from dwi_ml.training.utils.experiment import (
    add_mandatory_args_training_experiment)
from dwi_ml.training.utils.trainer import add_training_args, run_experiment, \
    format_lr


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_mandatory_args_training_experiment(p)
    add_logging_arg(p)
    add_args_batch_sampler(p)
    add_args_batch_loader(p)
    add_training_args(p)

    # Specific to Transformers:
    gt = add_abstract_model_args(p)
    add_tto_model_args(gt)

    add_memory_args(p, add_lazy_options=True, add_rng=True)

    return p


def init_from_args(args, sub_loggers_level):
    torch.manual_seed(args.rng)  # Set torch seed

    # Prepare the dataset
    dataset = prepare_multisubjectdataset(args, load_testing=False,
                                          log_level=sub_loggers_level)

    # Preparing the model
    # (general args)
    args, dg_args = perform_checks(args)
    # (nb features)
    input_group_idx = dataset.volume_groups.index(args.input_group_name)
    args.nb_features = dataset.nb_features[input_group_idx]
    # Final model
    with Timer("\n\nPreparing model", newline=True, color='yellow'):
        model = OriginalTransformerModel(
            experiment_name=args.experiment_name,
            step_size=args.step_size, compress=args.compress_lines,
            # Targets in decoder:
            token_type=args.token_type,
            # Concerning inputs:
            max_len=args.max_len, nb_features=args.nb_features,
            positional_encoding_key=args.position_encoding,
            embedding_key_x=args.data_embedding,
            embedding_key_t=args.target_embedding,
            # Torch's transformer parameters
            d_model=args.d_model, ffnn_hidden_size=args.ffnn_hidden_size,
            nheads=args.nheads, dropout_rate=args.dropout_rate,
            activation=args.activation, norm_first=args.norm_first,
            n_layers_e=args.n_layers_e, n_layers_d=args.n_layers_e,
            start_from_copy_prev=args.start_from_copy_prev,
            # Direction getter
            dg_key=args.dg_key, dg_args=dg_args,
            # Other
            neighborhood_type=args.neighborhood_type,
            neighborhood_radius=args.neighborhood_radius,
            log_level=sub_loggers_level)

        logging.info("Transformer (original) model final parameters:" +
                     format_dict_to_str(model.params_for_checkpoint))

    batch_sampler = prepare_batch_sampler(dataset, args, sub_loggers_level)
    batch_loader = prepare_batch_loader(dataset, model, args, sub_loggers_level)

    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        lr = format_lr(args.learning_rate)
        trainer = TransformerTrainer(
            model, args.experiments_path, args.experiment_name,
            batch_sampler, batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rates=lr, weight_decay=args.weight_decay,
            optimizer=args.optimizer, max_epochs=args.max_epochs,
            max_batches_per_epoch_training=args.max_batches_per_epoch_training,
            max_batches_per_epoch_validation=args.max_batches_per_epoch_validation,
            patience=args.patience, patience_delta=args.patience_delta,
            from_checkpoint=False,
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
    logging.getLogger().setLevel(level=args.logging)

    # Check that all files exist
    assert_inputs_exist(p, [args.hdf5_file])
    assert_outputs_exist(p, args, args.experiments_path)

    # Verify if a checkpoint has been saved. Else create an experiment.
    if os.path.exists(os.path.join(args.experiments_path, args.experiment_name,
                                   "checkpoint")):
        raise FileExistsError("This experiment already exists. Delete or use "
                              "script tto_resume_training_from_checkpoint.py.")

    trainer = init_from_args(args, sub_loggers_level)

    run_experiment(trainer)


if __name__ == '__main__':
    main()
