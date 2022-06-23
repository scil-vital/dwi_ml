# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.training.monitoring import EarlyStoppingError
from dwi_ml.experiment_utils.timer import Timer

logger = logging.getLogger('train_logger')


def add_training_args(p: argparse.ArgumentParser):
    training_group = p.add_argument_group("Training")
    training_group.add_argument(
        '--learning_rate', type=int, default=0.001, metavar='r',
        help="Learning rate. [0.001] (torch's default)")
    training_group.add_argument(
        '--weight_decay', type=float, default=0.01, metavar='v',
        help="Add a weight decay penalty on the parameters. [0.01] "
             "(torch's default).")
    training_group.add_argument(
        '--max_epochs', type=int, default=100, metavar='n',
        help="Maximum number of epochs. [100]")
    training_group.add_argument(
        '--patience', type=int, default=20, metavar='n',
        help="Use early stopping. Defines the number of epochs after which \n"
             "the model should stop if the loss hasn't improved. \n"
             "Default: same as max_epochs.")
    training_group.add_argument(
        '--max_batches_per_epoch_training', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch. This will help avoid long\n"
             "epochs, to ensure that we save checkpoints regularly. [1000]")
    training_group.add_argument(
        '--max_batches_per_epoch_validation', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch during validation.")

    comet_g = p.add_argument_group("Comet")
    comet_g.add_argument(
        '--comet_workspace', metavar='w',
        help='Your comet workspace. If not set, comet.ml will not be used.\n'
             'See our docs/Getting Started for more information on comet \n'
             'and its API key.')
    comet_g.add_argument(
        '--comet_project', metavar='p',
        help='Send your experiment to a specific comet.ml project. If not \n'
             'set, it will be sent to Uncategorized Experiments.')
    return training_group


def run_experiment(trainer):
    # Run (or continue) the experiment
    try:
        with Timer("\n****** Training and validating model!!! ********",
                   newline=True, color='magenta'):
            trainer.train_and_validate()
    except EarlyStoppingError as e:
        print(e)

    # Model already saved in the last checkpoint, but we could save it again.
    # trainer.model.save(trainer.saving_path)

    trainer.logger.info("Script terminated successfully. \n"
                        "Saved experiment in folder : {}"
                        .format(trainer.saving_path))
    trainer.logger.info("Summary: ran {} epochs (out of max {}). \n"
                        "Best loss was {} at epoch #{}"
                        .format(trainer.current_epoch + 1,
                                trainer.max_epochs,
                                trainer.best_epoch_monitoring.best_value,
                                trainer.best_epoch_monitoring.best_epoch + 1))
