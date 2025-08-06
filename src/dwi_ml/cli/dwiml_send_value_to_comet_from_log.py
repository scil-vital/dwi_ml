#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sends all values to comet from a log (expecting the log to be one value per
epoch). Note that the comet experiment is found by reading the experiment's
checkpoing (must exist).

USE WITH CARE. This cannot be undone.
"""
import argparse
import glob
import os
from time import sleep

import numpy as np
from comet_ml import ExistingExperiment
from scilpy.io.utils import assert_inputs_exist

from dwi_ml.training.trainers import DWIMLAbstractTrainer


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiments_path',
                   help="Path to the experiments folder.")
    p.add_argument('experiment_name',
                   help="Name of the experiment.")
    p.add_argument('--logs', nargs='+',
                   help="Log file (s). Expected to be located inside the "
                        "experiments_path. If not set, uses all logs.")
    p.add_argument('--metric_names', nargs='+',
                   help="Comet.ml's metric name(s). Must contain the same "
                        "number of inputs as --logs.\n"
                        "If not set, we will suggest you the probable name(s) "
                        "but we will not run the script.")
    p.add_argument('--use_suggested_name', action='store_true',
                   help="If set and --metric_name is not set, will run with "
                        "the suggested name(s).")
    p.add_argument('--use_best', action='store_true',
                   help="If set, uses only the best value in segment [0, t] "
                        "as value at time t. (Best = lowest).")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Verifications
    experiment_path = os.path.join(args.experiments_path, args.experiment_name)
    if not os.path.isdir(experiment_path):
        parser.error('Experiment does not exist: {}'.format(experiment_path))

    # Prepare logs, with possible wildcards.
    log_path = os.path.join(str(experiment_path), 'logs')
    if args.logs is None:
        log_paths = glob.glob(os.path.join(log_path, '*'))
    else:
        log_paths = [os.path.join(log_path, file) for file in args.logs]

    assert_inputs_exist(parser, log_paths)

    # Prepare associated metric names (if not given)
    if args.metric_names is None:
        args.metric_names = []
        if args.use_best:
            print("Not sure what to suggest you for --metric_name with "
                  "option --use_best. Probably 'best_loss'!?")
            args.use_suggested_name = False
        else:
            for log in log_paths:
                # Based on current implementation of things:
                # - comet adds train_ or validate_ as prefix
                # - we add _per_epoch.npy as suffix to the log name.
                base_name = os.path.basename(log)
                metric_name, _ = os.path.splitext(base_name)

                # Add comet prefix
                if 'tracking' in metric_name or 'valid' in metric_name:
                    metric_name = 'validate_' + metric_name
                elif 'training' in metric_name:
                    metric_name = 'train_' + metric_name

                # Remove our suffix
                if metric_name[-10:] == '_per_epoch':
                    metric_name = metric_name[:-10]

                print("Suggested --metric_name for log {}: {}"
                      .format(base_name, metric_name))
                args.metric_names.append(metric_name)

        # Possibly stop now
        if not args.use_suggested_name:
            return

    # Verify
    if not len(args.metric_names) == len(log_paths):
        parser.error("Expecting the same number of metrics_names (got {}) "
                     "than logs (got {})."
                     .format(len(args.metric_names), len(log_paths)))

    # Loading comet from info in checkpoint
    checkpoint_state = DWIMLAbstractTrainer.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)
    current_states = checkpoint_state['current_states']
    comet_key = current_states['comet_key']

    print("Found comet experiment: key {}. Loading.".format(comet_key))
    comet_exp = ExistingExperiment(previous_experiment=comet_key)

    for log, metric_name in zip(log_paths, args.metric_names):
        # Loading the log
        data = np.load(log)

        # Note. There is comet_exp.metrics with the metrics previously logged.
        # But seems to only contain metrics logged in current session.
        # Now, metrics={}.
        print("Will send values for all {} epochs to metric {}"
              .format(len(data), metric_name))
        best_value = np.inf
        for t in range(len(data)):
            value = data[t]
            if args.use_best:
                best_value = min(value, best_value)
                value = best_value

            # Send value
            comet_exp.log_metric(metric_name, value, step=t)

            # Rate limits:
            # https://www.comet.com/docs/v2/api-and-sdk/python-sdk/warnings-errors/
            # 10,000 per minute = 167 per second.
            # If we wait 0.01 second, max 100 call per second, will help.
            # Note. Sleep on Linux seems to allow 1ms sleep. On windows, could
            # fail. https://python-forum.io/thread-17019.html
            sleep(0.01)

    print("Done!\n\n")


if __name__ == '__main__':
    main()
