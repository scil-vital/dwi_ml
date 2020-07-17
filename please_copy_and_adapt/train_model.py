#!/usr/bin/env python
# -*- coding: utf-8 -*-

###########################
# Remove or add parameters to fit your needs. You should change your yaml file
# accordingly.
# Change DWIMLAbstractSequences for an implementation of your own model.
# It should be a child of this abstract class.
############################

""" Train a model for my favorite experiment"""

import argparse
import logging
from os import path
import yaml

from dwi_ml.training.checks_for_experiment_parameters import (
    check_all_experiment_parameters, check_logging_level)
from dwi_ml.training.trainer_abstract import DWIMLAbstractSequences


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('parameters_filename',
                   help='Experiment configuration YAML filename. See '
                        'please_copy_and_adapt/training_parameters.yaml for an '
                        'example.')

    arguments = p.parse_args()
    return arguments


def main():
    args = parse_args()

    # Load parameters from yaml file
    if not path.exists(args.parameters_filename):
        raise FileNotFoundError("Yaml file not found: "
                                "{}".format(args.parameters_filename))
    with open(args.parameters_filename) as f:
        conf = yaml.safe_load(f.read())

    # Initialize logger
    logging_level = check_logging_level(conf['logging']['level'],
                                        required=False)
    logging.basicConfig(level=logging_level)
    logging.info(conf)

    # Perform checks.checks
    organized_args = check_all_experiment_parameters(conf)

    # Instantiate your class
    # (Change DWIMLAbstractSequences for your class.)
    # Then load dataset, build model, train and save
    experiment = DWIMLAbstractSequences(organized_args)
    experiment.load_dataset()
    experiment.build_model()
    experiment.train()
    experiment.save()


if __name__ == '__main__':
    main()
