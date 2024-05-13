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
             "[lr*step]. For instance, \n"
             "--learning_rate 0.001*3 0.0001 would set the lr to 0.001 "
             "for the 3 first epochs, \nand 0.0001 for the remaining epochs."
             "(torch's default = 0.001)")
    training_group.add_argument(
        '--weight_decay', type=float, default=0.01, metavar='v',
        help="Add a weight decay penalty on the parameters (regularization ) "
             "(torch's default: \n0.01).")
    training_group.add_argument(
        '--optimizer', choices=['Adam', 'RAdam', 'SGD'], default='Adam',
        help="Choice of torch optimizer amongst ['Adam', 'RAdam', 'SGD']. "
             "Default: Adam.")
    training_group.add_argument(
        '--max_epochs', type=int, default=100, metavar='n',
        help="Maximum number of epochs. [100]")
    training_group.add_argument(
        '--patience', type=int, metavar='n',
        help="If set, uses early stopping. Defines the number of epochs after "
             "which the model \nshould stop if the loss hasn't improved.")
    training_group.add_argument(
        '--patience_delta', type=float, default=1e-6, metavar='eps',
        help="Limit difference between two validation losses to consider that "
             "the model has \nimproved between two epochs. [1e-6]")
    training_group.add_argument(
        '--max_batches_per_epoch_training', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch. This will help avoid long "
             "epochs, to ensure \nthat we save checkpoints regularly. [1000]")
    training_group.add_argument(
        '--max_batches_per_epoch_validation', type=int, default=1000,
        metavar='n',
        help="Maximum number of batches per epoch during validation.")
    training_group.add_argument(
        '--clip_grad', type=float, default=None,
        help="Value to which the gradient norms to avoid exploding gradients."
             "\nDefault = None (not clipping).")

    if add_a_tracking_validation_phase:
        training_group.add_argument(
            '--add_a_tracking_validation_phase', action='store_true',
            help="If set, a generation validation phase (GV) will be added.")
        training_group.add_argument(
            '--tracking_phase_frequency', type=int, default=1, metavar='N',
            help="The GV phase can be computed at every epoch (default), or "
                 "once every N epochs.")
        training_group.add_argument(
            '--tracking_mask',
            help="Volume group to use as tracking mask during the generation "
                 "phase.")
        training_group.add_argument(
            '--tracking_phase_nb_segments_init', type=int, default=1,
            metavar='N',
            help="Number of segments copied from the 'real' validation "
                 "streamlines before starting \npropagation during GV phases "
                 "[1].")

    comet_g = p.add_argument_group("Comet")
    comet_g.add_argument(
        '--comet_workspace', metavar='w',
        help='Your comet workspace. If not set, comet.ml will not be used. '
             'See our doc for more \ninformation on comet and its API key: \n'
             'https://dwi-ml.readthedocs.io/en/latest/getting_started.html')
    comet_g.add_argument(
        '--comet_project', metavar='p',
        help='Send your experiment to a specific comet.ml project. If not '
             'set, it will be sent \nto Uncategorized Experiments.')
    return training_group


def format_lr(lr_arg):
    """
    Formatting [0.001*2 0.002] into [0.001 0.001 0.002].
    """
    if lr_arg is None or isinstance(lr_arg, float):
        return lr_arg

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
                             "float*int: learning_rate*nb_epochs.")

    try:
        all_lr += [float(lr_arg[-1])]
    except ValueError:
        raise ValueError("The list of learning rates should end with a final "
                         "float that will be kept fixed until the end of "
                         "training.")

    if len(all_lr) == 1:
        return all_lr[0]

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
