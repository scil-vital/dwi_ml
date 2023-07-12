# -*- coding: utf-8 -*-
import argparse
import logging

from dwi_ml.training.utils.monitoring import EarlyStoppingError
from dwi_ml.experiment_utils.timer import Timer

logger = logging.getLogger('train_logger')


def get_training_args(add_a_tracking_validation_phase=False):
    args = {
        '--learning_rate': {
            'metavar': 'r', 'nargs': '+', 'type': float, 'default': 0.001,
            'help': "Learning rate. Basic usage: a single float value. "
                    "(torch's 'default' : 0.001) \n"
                    "Can also be a list of [lr * nb_epochs, final_lr]. Ex: "
                    "'--learning_rate 0.001*3 0.0001' \nwould set the lr to "
                    "0.001 for the first 3 epochs, and to 0.0001 for the "
                    "remaining epochs."},
        '--lr_decrease_params': {
            'metavar': 'E L', 'nargs': 2, 'type': float,
            'help': "Parameters [E, L] to set the learning rate an exponential "
                    "decreasing curve. \nThe final curve will be "
                    "init_lr * exp(-x / r). The rate of \ndecrease, r, is "
                    "defined in order to ensure that the learning rate curve "
                    "will hit \nvalue L at epoch E.\n"
                    "learning_rate must be a single float value."},
        '--weight_decay': {
            'type': float, 'default': 0.01, 'metavar': 'v',
            'help': "Adds a weight decay penalty on the parameters "
                    "(regularization parameter) \n(torch's default: 0.01)."},
        '--optimizer': {
            'choices': ['Adam', 'RAdam', 'SGD'], 'default': 'Adam',
            'help': "Choice of torch optimizer amongst ['Adam', 'RAdam', "
                    "'SGD']. Default: Adam."},
        '--max_epochs': {
            'type': int, 'default': 100, 'metavar': 'n',
            'help': "Maximum number of epochs. Default: 100, for no good "
                    "reason."},
        '--patience': {
            'type': int, 'default': 20, 'metavar': 'n',
            'help': "Use early stopping. Defines the number of epochs after "
                    "which the model \nshould stop if the loss hasn't "
                    "improved. Default: 20, for no good reason."},
        '--patience_delta': {
            'type': float, 'default': 1e-6, 'metavar': 'eps',
            'help': "Limit difference between two validation losses to consider "
                    "that the model \nimproved between the two epochs."},
        '--max_batches_per_epoch_training': {
            'type': int, 'default': 1000, 'metavar': 'n',
            'help': "Maximum number of batches per epoch. This will 'help' "
                    "avoid long epochs, \nto ensure that we save checkpoints "
                    "regularly. Default: 1000, for no good reason."},
        '--max_batches_per_epoch_validation': {
            'type': int, 'default': 1000, 'metavar': 'n',
            'help': "Idem, for validation."}
    }

    if add_a_tracking_validation_phase:
        args.update({
            '--add_a_tracking_validation_phase': {
                'action': 'store_true',
                'help': "Adds a generation part to the validation phase: "
                        "generates streamline starting at \nthe validation "
                        "batch's starting point, and compare ending points "
                        "with expected \nvalues."},
            '--tracking_phase_frequency': {
                'type': int, 'default': 1,
                'help': "The tracking-validation phase can be done every N "
                        "epochs. Default: 1."},
            '--tracking_mask': {
                'help': "Volume group to use as tracking mask during the "
                        "generation phase. Required if \n"
                        "--add_a_tracking_validation_phase is set."},
            '--tracking_phase_nb_segments_init': {
                'type': int, 'default': 5,
                'help': "Number of segments copied from the 'real' streamlines "
                        "before starting \npropagation during generation phases."}
        })
    return args


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
