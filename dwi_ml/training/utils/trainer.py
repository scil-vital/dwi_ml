# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.training.utils.monitoring import EarlyStoppingError
from dwi_ml.experiment_utils.timer import Timer

logger = logging.getLogger('train_logger')


def add_training_args(p: argparse.ArgumentParser,
                      add_a_tracking_validation_phase=False):
    training_group = p.add_argument_group("Training")
    training_group.add_argument(
        '--learning_rate', metavar='r', nargs='+',
        help="Learning rate. Can be set as a single float, or as a list of "
             "[lr*step]. \n"
             "Ex: '--learning_rate 0.001*3 0.0001' would set the lr to 0.001 "
             "for the first \n3 epochs, and 0.0001 for the remaining epochs.\n"
             "(torch's default = 0.001)")
    training_group.add_argument(
        '--weight_decay', type=float, default=0.01, metavar='v',
        help="Add a weight decay penalty on the parameters (regularization "
             "parameter)\n[0.01] (torch's default).")
    training_group.add_argument(
        '--optimizer', choices=['Adam', 'RAdam', 'SGD'], default='Adam',
        help="Choice of torch optimizer amongst ['Adam', 'RAdam', 'SGD'].\n"
             "Default: Adam.")
    training_group.add_argument(
        '--max_epochs', type=int, default=100, metavar='n',
        help="Maximum number of epochs. [100]")
    training_group.add_argument(
        '--patience', type=int, default=20, metavar='n',
        help="Use early stopping. Defines the number of epochs after which \n"
             "the model should stop if the loss hasn't improved. \n"
             "Default: same as max_epochs.")
    training_group.add_argument(
        '--patience_delta', type=float, default=1e-6, metavar='eps',
        help="Limit difference between two validation losses to consider that "
             "\nthe model improved between the two epochs.")
    training_group.add_argument(
        '--max_batches_per_epoch_training', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch. This will help avoid long\n"
             "epochs, to ensure that we save checkpoints regularly. [1000]")
    training_group.add_argument(
        '--max_batches_per_epoch_validation', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch during validation.")

    if add_a_tracking_validation_phase:
        training_group.add_argument(
            '--add_a_tracking_validation_phase', action='store_true')
        training_group.add_argument(
            '--tracking_phase_frequency', type=int, default=5)
        training_group.add_argument(
            '--tracking_mask',
            help="Volume group to use as tracking mask during the generation "
                 "phase.")
        training_group.add_argument(
            '--tracking_phase_nb_steps_init', type=int, default=5,
            help="Number of segments copied from the 'real' streamlines "
                 "before starting propagation during generation phases.")

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


def format_lr(lr_arg):
    """
    Formatting [0.001*2 0.002] into [0.001 0.001 0.002].
    """
    if lr_arg is None:
        return None

    all_lr = []
    for lr in lr_arg[:-1]:
        assert '*' in lr, "When using multiple learning rates, you should " \
                          "define the number of epochs to use each, with " \
                          "the notation lr*nb_epochs."
        _lr, nb = lr.split('*')
        try:
            all_lr += [float(_lr)] * int(nb)
        except ValueError:
            raise ValueError("The first learning rates should be formatted a "
                             "float*int: learning_rate*nb_epoch.")

    try:
        all_lr += [float(lr_arg[-1])]
    except ValueError:
        raise ValueError("The list of learning rates should end with a final "
                         "float that will be kept fixed until the end of "
                         "training.")
    return all_lr


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

    logger.info("Script terminated successfully. \n"
                "Saved experiment in folder : {}".format(trainer.saving_path))
    logger.info("Summary: ran {} epochs (out of max {}). \n"
                "Best loss was {} at epoch #{}.\n"
                .format(trainer.current_epoch + 1, trainer.max_epochs,
                        trainer.best_epoch_monitor.best_value,
                        trainer.best_epoch_monitor.best_epoch + 1))
