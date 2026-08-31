#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Print the total training time / validation time
"""
import argparse
import csv
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from scilpy.io.utils import (add_overwrite_arg, add_verbose_arg,
                             assert_outputs_exist)

from dwi_ml.general.viz.logs_plots import visualize_logs


def _build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("experiment",
                   help="Path to the experiment folder.")

    add_verbose_arg(p)
    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    logging.getLogger().setLevel(level=args.verbose)

    # ---------------
    # Verifications
    # ---------------
    if not pathlib.Path(args.experiment).exists():
        raise ValueError("Experiment folder does not exist: {}"
                         .format(args.experiment))
    logs_path = pathlib.Path(args.experiment, 'logs')
    if not logs_path.exists():
        raise ValueError("Logs folder does not exist for experiment {}!"
                         .format(args.experiment))

    # Loading
    training_times = np.load(os.path.join(logs_path, 'training_time_monitor_duration.npy'))
    validation_times = np.load(os.path.join(logs_path, 'validation_time_monitor_duration.npy'))

    print("Number of training epochs: {}".format(len(training_times)))
    print("Number of validation epochs: {}\n".format(len(validation_times)))

    time_training = sum(training_times)
    time_validation = sum(validation_times)
    print("Total time training: {}".format(time_training))
    print("Total time validation: {}\n".format(time_validation))

    print("Total time all together: {:.2f} minutes, i.e. {:.2f} hours, "
          "i.e. {:.2f} days."
          .format(time_training + time_validation,
                  (time_training + time_validation)/60,
                  (time_training + time_validation)/60 /24))

if __name__ == '__main__':
    main()
