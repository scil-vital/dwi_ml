#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import torch
import numpy as np
from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram
from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                             add_memory_args)
from dwi_ml.models.projects.ae_models import ModelAE
from dwi_ml.viz.latent_streamlines import BundlesLatentSpaceVisualizer
from dipy.tracking.streamline import set_number_of_points


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)
    # Mandatory
    # Should only be False for debugging tests.
    add_arg_existing_experiment_path(p)
    # Add_args_testing_subj_hdf5(p)

    p.add_argument('in_tractogram',
                   help="If set, saves the tractogram with the loss per point "
                        "as a data per point (color)")

    p.add_argument('out_tractogram',
                   help="If set, saves the tractogram with the loss per point "
                        "as a data per point (color)")
    p.add_argument('--viz_save_path', type=str, default=None,
                   help="Path to save the figure. If not specified, the figure will be shown.")

    # Options
    p.add_argument('--batch_size', type=int)
    add_memory_args(p)

    p.add_argument('--pick_at_random', action='store_true')
    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


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
    assert_outputs_exist(p, args, args.out_tractogram)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load model
    logging.debug("Loading model.")
    model = ModelAE.load_model_from_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_loggers_level).to(device)

    # Setup vizualisation
    ls_viz = BundlesLatentSpaceVisualizer(save_path=args.viz_save_path)
    model.register_hook_post_encoding(
        lambda encoded_data: ls_viz.add_data_to_plot(encoded_data))

    sft = load_tractogram_with_reference(p, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()
    bundle = sft.streamlines[0:5000]

    logging.info("Running model to compute loss")

    new_sft = sft.from_sft(bundle, sft)
    save_tractogram(new_sft, 'orig_5000.trk')

    with torch.no_grad():
        streamlines = torch.as_tensor(np.asarray(set_number_of_points(bundle, 256)),
                                      dtype=torch.float32, device=device)
        tmp_outputs = model(streamlines)

        ls_viz.plot()

    streamlines_output = [tmp_outputs[i, :, :].transpose(
        0, 1).cpu().numpy() for i in range(len(bundle))]
    new_sft = sft.from_sft(streamlines_output, sft)
    save_tractogram(new_sft, args.out_tractogram, bbox_valid_check=False)


if __name__ == '__main__':
    main()
