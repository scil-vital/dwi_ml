# -*- coding: utf-8 -*-
import json
import logging
import os
from argparse import ArgumentParser
from typing import List

import torch

from dwi_ml.data.dataset.multi_subject_containers import (MultiSubjectDataset,
                                                          MultisubjectSubset)


def add_args_testing_subj_hdf5(p: ArgumentParser, optional_hdf5=False,
                               ask_input_group=False,
                               ask_streamlines_group=False):
    g = p.add_argument_group("Inputs options")
    if optional_hdf5:
        g.add_argument('--hdf5_file', metavar='file',
                       help="Path to the hdf5 file. If not given, will use "
                            "the file from the experiment's \nparameters. "
                            "(in parameters_latest.json)")
    else:
        p.add_argument('hdf5_file',
                       help="Path to the hdf5 file.")
    p.add_argument('subj_id',
                   help="Subject id to use in the hdf5.")
    if ask_input_group:
        p.add_argument('input_group',
                       help="Model's input's volume group in the hdf5.")
    if ask_streamlines_group:
        p.add_argument('streamlines_group',
                       help="Model's streamlines group in the hdf5.")
    g.add_argument('--subset', default='testing',
                   choices=['training', 'validation', 'testing'],
                   help="Subject id should probably come come the "
                        "'testing' set but you can modify this \nto "
                        "'training' or 'validation'.")


def find_hdf5_associated_to_experiment(experiment_path):
    parameters_json = os.path.join(experiment_path, 'parameters_latest.json')
    hdf5_file = None
    if os.path.isfile(parameters_json):
        with open(parameters_json, 'r') as json_file:
            params = json.load(json_file)
            if 'hdf5 file' in params:
                hdf5_file = params['hdf5 file']

    if hdf5_file is None:
        logging.warning("Did not find the hdf5 file associated to your "
                        "exeperiment in the parameters file {}.\n"
                        "Will try to find it in the latest checkpoint."
                        .format(parameters_json))
        checkpoint_path = os.path.join(
            experiment_path, "checkpoint", "checkpoint_state.pkl")
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                'Checkpoint was not found! ({}). Could not find the hdf5 '
                'associated to your experiment. Please specify it yourself.'
                .format(checkpoint_path))
        else:
            checkpoint_state = torch.load(checkpoint_path)
            hdf5_file = checkpoint_state['dataset_params']['hdf5_file']

    return hdf5_file


def prepare_dataset_one_subj(
        hdf5_file: str, subj_id: str, lazy: bool = False, cache_size: int = 1,
        subset_name: str = 'testing', volume_groups: List[str] = None,
        streamline_groups: List[str] = None,
        log_level=None) -> MultisubjectSubset:
    """
    Loading a MultiSubjectDataset with only one subject.
    """
    dataset = MultiSubjectDataset(hdf5_file, lazy=lazy, cache_size=cache_size,
                                  log_level=log_level)

    possible_subsets = ['training', 'validation', 'testing']
    if subset_name not in possible_subsets:
        raise ValueError("Subset name should be one of {}"
                         .format(possible_subsets))
    load_training = True if subset_name == 'training' else False
    load_validation = True if subset_name == 'validation' else False
    load_testing = True if subset_name == 'testing' else False

    dataset.load_data(load_training, load_validation, load_testing,
                      subj_id, volume_groups, streamline_groups)

    if subset_name == 'testing':
        subset = dataset.testing_set
    elif subset_name == 'training':
        subset = dataset.training_set
    elif subset_name == 'validation':
        subset = dataset.validation_set
    else:
        raise ValueError("Subset must be one of 'training', 'validation' "
                         "or 'testing.")

    return subset
