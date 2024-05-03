#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Copies an existing experiment to another folder, updating deprecated values.
"""
import argparse
import json
import logging
import os
import shutil

import numpy as np
import torch
from scilpy.io.utils import add_overwrite_arg

from dwi_ml.data.dataset.utils import prepare_multisubjectdataset
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.io_utils import add_verbose_arg
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.projects.learn2track_trainer import Learn2TrackTrainer
from dwi_ml.training.with_generation.batch_loader import \
    DWIMLBatchLoaderWithConnectivity


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('experiment_path',
                   help='Name for the experiment.')
    p.add_argument('out_experiment',
                   help="Name of the fixed experiment.")
    p.add_argument('--new_hdf5',
                   help="Required only if previous hdf5 has been moved.")

    add_overwrite_arg(p)
    add_verbose_arg(p)

    return p


def replace(params, old_key, new_key):
    if old_key in params:
        params[new_key] = params[old_key]
        del params[old_key]
    else:
        logging.warning("Expected to find deprecated key {} (to be replaced "
                        "by {}), but did not find it. Skipping."
                        .format(old_key, new_key))
    return params


def fix_deprecated_model_params(params):
    # embedding_size --> embedded_size
    params = replace(params, 'prev_dirs_embedding_size',
                     'prev_dirs_embedded_size')
    params = replace(params, 'input_embedding_size',
                     'input_embedded_size')

    # deleted size_ratio option
    if 'input_embedding_size_ratio' in params:
        assert params['input_embedding_size_ratio'] is None, \
            ("Can't fix deprecated value 'inpute_embedded_size_ratio'; has no"
             " new equivalent. I thought I never used it.")
        del params['input_embedding_size_ratio']

    # Added cnn options
    if 'nb_cnn_filters' not in params:
        params['nb_cnn_filters'] = None
    if 'kernel_size' not in params:
        params['kernel_size'] = None

    # Neighborhood management modified
    r = params['neighborhood_radius']
    if isinstance(r, list):
        params['neighborhood_radius'] = len(r)
        params['neighborhood_resolution'] = r[0]
        if len(r) > 1:
            if not np.all(np.diff(r) == r[0]):
                raise ValueError("Now, neighborhood must have the same "
                                 "resolution between each layer of "
                                 "neighborhood. But got: {}".format(r))

    return params


def fix_deprecated_checkpoint_params(checkpoint_state):
    # 1) Dataset params: better use a --new_hdf5 to fix.

    # 2) Trainer params : nb_steps --> nb_segments
    checkpoint_state['params_for_init'] = replace(
        checkpoint_state['params_for_init'], 'tracking_phase_nb_steps_init',
        'tracking_phase_nb_segments_init')

    # 3) Monitors: new var ever_max, ever_min
    for k in checkpoint_state['current_states'].keys():
        if isinstance(checkpoint_state['current_states'][k], dict) and \
                'average_per_epoch' in checkpoint_state['current_states'][k].keys():
            # Found a monitor
            if 'ever_min' not in checkpoint_state['current_states'][k]:
                logging.warning("Setting false min, max for monitor {}. "
                                # But not sure this is ever used.... TODO
                                .format(k))
                checkpoint_state['current_states'][k]['ever_min'] = -np.inf
                checkpoint_state['current_states'][k]['ever_max'] = np.inf

    # 4) unclipped_grad_norm_monitor!
    if 'unclipped_grad_norm_monitor_state' not in \
            checkpoint_state['current_states']:
        logging.warning("Copy grad norm monitor as fake unclipped grad norm "
                        "monitor")
        checkpoint_state['current_states']['unclipped_grad_norm_monitor_state'] = \
            checkpoint_state['current_states']['grad_norm_monitor_state']

    return checkpoint_state


def fix_both_model_parameter_json_files(args):
    # 1) Loading params from checkpoint's model
    model_dir = os.path.join(args.experiment_path, 'checkpoint', 'model')
    params = Learn2TrackModel._load_params(model_dir)

    # 2) Loading params from best model and verifying that they fit
    model_dir = os.path.join(args.experiment_path, 'best_model')
    params2 = Learn2TrackModel._load_params(model_dir)
    assert params == params2, ("Unexpected error. Parameters in the "
                               "checkpoint dir and in the best_model dir "
                               "should be the same. Did you modify the "
                               "parameters.json files?\n"
                               "Checkpoint params: \n"
                               "{}\n"
                               "--------------------"
                               "Best model params: \n"
                               "{}"
                               .format(format_dict_to_str(params),
                                       format_dict_to_str(params2)))
    del params2
    logging.debug("Loaded params:\n{}".format(format_dict_to_str(params)))

    # 3) Fixing params
    params = fix_deprecated_model_params(params)
    print("\n\n----------------Fixed the model parameters ----------------\n"
          "Reformated model's params:\n " + format_dict_to_str(params))

    # 4) Save fixed params in both parameters files
    fixed_checkpoint_model_dir = os.path.join(
        args.out_experiment, "checkpoint", "model")
    params_in_checkpoint = os.path.join(
        fixed_checkpoint_model_dir, "parameters.json")
    with open(params_in_checkpoint, 'w') as json_file:
        json_file.write(json.dumps(params, indent=4, separators=(',', ': ')))
    fixed_best_model_dir = os.path.join(args.out_experiment, "best_model")
    params_in_best_model = os.path.join(fixed_best_model_dir,
                                        "parameters.json")
    with open(params_in_best_model, 'w') as json_file:
        json_file.write(json.dumps(params, indent=4, separators=(',', ': ')))

    # Verify that both models can be loaded
    _ = Learn2TrackModel.load_model_from_params_and_state(
        fixed_checkpoint_model_dir)
    model = Learn2TrackModel.load_model_from_params_and_state(
        fixed_best_model_dir)

    return model


def fix_checkpoint(args, model):
    # Fixing trainer

    # Loading checkpoint
    experiments_path, experiment_name = os.path.split(args.experiment_path)
    checkpoint_state = Learn2TrackTrainer.load_params_from_checkpoint(
        experiments_path, experiment_name)

    # Verify hdf5
    dataset_params = checkpoint_state['dataset_params']['training set']
    if not os.path.isfile(dataset_params['hdf5_file']):
        if args.new_hdf5 is None:
            raise ValueError("hdf5 file has been deleted or moved ({})\n"
                             "Please set a path to a new hdf5.")
        else:
            # Get the hdf5
            dataset = prepare_multisubjectdataset(
                argparse.Namespace(**{'hdf5_file': args.new_hdf5,
                                      'lazy': True,
                                      'cache_size': 1}))
            # Compare all values
            for k, v in dataset_params.items():
                if k not in ['set_name', 'hdf5_file', 'lazy']:
                    assert dataset.training_set.__getattribute__(k) == v, \
                        ("Value {} in old hdf5 (training set) was {} but is "
                         "{} in the new one!"
                         .format(k, v,
                                 dataset.training_set.__getattribute__(k)))
                    assert dataset.validation_set.__getattribute__(k) == v, \
                        ("Value {} in old hdf5 (validation set) was {} but is "
                         "{} in the new one!"
                         .format(k, v,
                                 dataset.training_set.__getattribute__(k)))

    elif args.new_hdf5 is not None:
        raise ValueError("We already have all required information from the "
                         "hdf5 at {}. We do not need a --new_hdf5.")
    else:
        # Ensure it was lazy
        dataset_params['lazy'] = True
        dataset = prepare_multisubjectdataset(
            argparse.Namespace(**dataset_params))

    # Fixing checkpoint
    checkpoint_state = fix_deprecated_checkpoint_params(checkpoint_state)
    checkpoint_dir = os.path.join(args.out_experiment, "checkpoint")
    torch.save(checkpoint_state,
               os.path.join(checkpoint_dir, "checkpoint_state.pkl"))

    # Init stuff will succeed if ok.
    batch_sampler = DWIMLBatchIDSampler.init_from_checkpoint(
        dataset, checkpoint_state['batch_sampler_params'])
    batch_loader = DWIMLBatchLoaderWithConnectivity.init_from_checkpoint(
            dataset, model, checkpoint_state['batch_loader_params'])
    experiments_path, experiment_name = os.path.split(args.out_experiment)
    trainer = Learn2TrackTrainer.init_from_checkpoint(
        model, experiments_path, experiment_name,
        batch_sampler, batch_loader,
        checkpoint_state, new_patience=None, new_max_epochs=None,
        log_level='WARNING')


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # General logging (ex, scilpy: Warning)
    logging.getLogger().setLevel(level=logging.WARNING)

    # Verify if a checkpoint has been saved.
    if not os.path.exists(args.experiment_path):
        raise FileNotFoundError("Experiment not found ({})."
                                .format(args.experiment_path))
    if os.path.exists(args.out_experiment):
        if args.overwrite:
            logging.warning("Careful. Deleting whole dir: {}"
                            .format(args.out_experiment))
            shutil.rmtree(args.out_experiment)
        else:
            raise FileExistsError("Out experiment already exists! ({})."
                                  .format(args.out_experiment))
    shutil.copytree(args.experiment_path, args.out_experiment)

    model = fix_both_model_parameter_json_files(args)
    fix_checkpoint(args, model)

    print("Out experiment {} should now be usable!"
          .format(args.out_experiment))


if __name__ == '__main__':
    main()
