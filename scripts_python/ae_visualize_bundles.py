#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import pathlib
import torch
import numpy as np
from glob import glob
from os.path import expanduser
from dipy.tracking.streamline import set_number_of_points

from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                           add_memory_args)
from dwi_ml.models.projects.ae_models import ModelAE
from dwi_ml.viz.latent_streamlines import BundlesLatentSpaceVisualizer


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Mandatory
    # Should only be False for debugging tests.
    add_arg_existing_experiment_path(p)
    # Add_args_testing_subj_hdf5(p)

    p.add_argument('in_bundles',
                   help="The 'glob' path to several bundles identified by their file name."
                   "e.g. FiberCupGroundTruth_filtered_bundle_0.tck")

    # Options
    p.add_argument('--batch_size', type=int)
    add_memory_args(p)

    p.add_argument('--pick_at_random', action='store_true')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p

def load_bundles(p, args, files_list: list):
    bundles = []
    for bundle_file in files_list:
        bundle_sft = load_tractogram_with_reference(p, args, bundle_file)
        bundle_sft.to_vox()
        bundle_sft.to_corner()
        bundles.append(bundle_sft)
    return bundles

def main():
    p = _build_arg_parser()
    args = p.parse_args()

    # Setting log level to INFO maximum for sub-loggers, else it becomes ugly,
    # but we will set trainer to user-defined level.
    sub_loggers_level = args.verbose if args.verbose != 'DEBUG' else 'INFO'

    # General logging (ex, scilpy: Warning)
    logging.getLogger().setLevel(level=logging.WARNING)

    # Verify output names
    # Check experiment_path exists and best_model folder exists
    # Assert_inputs_exist(p, args.hdf5_file)
    assert_outputs_exist(p, args, [])

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load model
    logging.debug("Loading model.")
    model = ModelAE.load_model_from_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_loggers_level).to(device)

    expanded = expanduser(args.in_bundles)
    bundles_files = glob(expanded)
    if isinstance(bundles_files, str):
        bundles_files = [bundles_files]

    bundles_label = [pathlib.Path(l).stem for l in bundles_files]
    bundles_sft = load_bundles(p, args, bundles_files)

    logging.info("Running model to compute loss")

    ls_viz = BundlesLatentSpaceVisualizer(
        save_path="/home/local/USHERBROOKE/levj1404/Documents/dwi_ml/data/out.png"
    )

    with torch.no_grad():
        for i, bundle_sft in enumerate(bundles_sft):
            
            # Resample
            streamlines = torch.as_tensor(np.asarray(set_number_of_points(bundle_sft.streamlines, 256)),
                                          dtype=torch.float32, device=device)
            
            latent_streamlines = model.encode(streamlines).cpu().numpy() # output of (N, 32)
            ls_viz.add_data_to_plot(latent_streamlines, label=bundles_label[i])

    ls_viz.plot()


if __name__ == '__main__':
    main()
