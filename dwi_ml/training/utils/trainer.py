# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.training.monitoring import EarlyStoppingError
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.experiment_utils.timer import Timer
from dwi_ml.training.trainers import DWIMLAbstractTrainer


def add_training_args(p: argparse.ArgumentParser):
    training_group = p.add_argument_group("Training")
    training_group.add_argument(
        '--learning_rate', type=int, default=0.001, metavar='r',
        help="Learning rate. Default: 0.001 (torch's default)")
    training_group.add_argument(
        '--weight_decay', type=float, default=0.01, metavar='v',
        help="Add a weight decay penalty on the parameters. Default: 0.01. "
             "(torch's default).")
    training_group.add_argument(
        '--max_epochs', type=int, default=100, metavar='n',
        help="Maximum number of epochs. Default: 100")
    training_group.add_argument(
        '--patience', type=int, default=20, metavar='n',
        help="Use early stopping. Defines the number of epochs after which "
             "the model should \nstop if the loss hasn't improved. "
             "Default: max_epochs.")
    training_group.add_argument(
        '--max_batches_per_epoch', type=int, default=1000, metavar='n',
        help="Maximum number of batches per epoch. This will help avoid "
             "long epochs, \nto ensure that we save checkpoints regularly.\n"
             "Default: 1000.")

    comet_g = p.add_argument_group("Comet")
    comet_g.add_argument(
        '--comet_workspace', metavar='w',
        help='Your comet workspace. If not set, comet.ml will not '
             'be used. See our \ndocs/Getting Started for more '
             'information on comet and its API key.')
    comet_g.add_argument(
        '--comet_project', metavar='p',
        help='Send your experiment to a specific comet.ml project. '
             'If not set, it will \nbe sent to Uncategorized '
             'Experiments.')
    return training_group


def prepare_trainer(training_batch_sampler, validation_batch_sampler,
                    training_batch_loader, validation_batch_loader,
                    model, args):
    # Instantiate trainer
    with Timer("\n\nPreparing trainer", newline=True, color='red'):
        trainer = DWIMLAbstractTrainer(
            model, args.experiment_path, args.experiment_name,
            training_batch_sampler, training_batch_loader,
            validation_batch_sampler, validation_batch_loader,
            # COMET
            comet_project=args.comet_project,
            comet_workspace=args.comet_workspace,
            # TRAINING
            learning_rate=args.learning_rate, max_epochs=args.max_epochs,
            max_batches_per_epoch=args.max_batches_per_epoch,
            patience=args.patience, from_checkpoint=False,
            weight_decay=args.weight_decay,
            # MEMORY
            # toDo check this
            nb_cpu_processes=args.processes,
            taskman_managed=args.taskman_managed, use_gpu=args.use_gpu)
        logging.info("Trainer params : " + format_dict_to_str(trainer.params))


def run_experiment(trainer, logging_choice):
    # Run (or continue) the experiment
    try:
        with Timer("\n\n****** Training and validating model!!! ********",
                   newline=True, color='magenta'):
            trainer.train_and_validate(logging_choice)
    except EarlyStoppingError as e:
        print(e)

    trainer.save_model()

    logging.info("Script terminated successfully. \n"
                 "Saved experiment in folder : {}"
                 .format(trainer.experiment_path))
    print("Summary: ran {} epochs. Best loss was {} at epoch #{}"
          .format(trainer.current_epoch + 1,
                  trainer.best_epoch_monitoring.best_value,
                  trainer.best_epoch_monitoring.best_epoch + 1))
