#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sends a value to comet. Note that the comet experiment is found by
reading the experiment's checkpoing (must exist).

USE WITH CARE. This cannot be undone.
"""
import argparse

from comet_ml import ExistingExperiment

from dwi_ml.training.trainers import DWIMLTrainer


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiments_path',
                   help="Path to the experiments folder.")
    p.add_argument('experiment_name',
                   help="Name of the experiment.")
    p.add_argument('metric_name',
                   help="Comet.ml's metric name.")
    p.add_argument('epoch', type=int,
                   help="Epoch to which value is associated.")
    p.add_argument('value', type=float,
                   help="Value to set.")

    return p


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Loading checkpoint
    checkpoint_state = DWIMLTrainer.load_params_from_checkpoint(
        args.experiments_path, args.experiment_name)

    current_states = checkpoint_state['current_states']
    comet_key = current_states['comet_key']

    comet_exp = ExistingExperiment(previous_experiment=comet_key)

    comet_exp.log_metric(args.metric_name, args.value, step=args.epoch)
    print("Sent metric {}: value = {} at epoch {}"
          .format(args.metric_name, args.value, args.epoch))


if __name__ == '__main__':
    main()
