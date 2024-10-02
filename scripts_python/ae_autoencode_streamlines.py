#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import nibabel as nib
import numpy as np
import torch

from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.tracking.utils import save_tractogram
from dipy.tracking.streamline import set_number_of_points
from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                             add_memory_args)
from dwi_ml.models.projects.ae_models import ModelAE
from dwi_ml.models.projects.ae_next_models import ModelConvNextAE
from nibabel.streamlines import detect_format


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

    # Additional arg for projects
    p.add_argument('--model', type=str, choices=['finta', 'convnext'],
                   default='finta',
                   help='Type of model to train')

    # Options
    p.add_argument('--normalize', action='store_true')
    p.add_argument('--batch_size', type=int, default=5000)
    add_memory_args(p)

    add_reference_arg(p)
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def autoencode_streamlines(
    model, sft, device, batch_size=5000, normalize=False
):
    sft.to_vox()
    sft.to_corner()

    bundle = set_number_of_points(sft.streamlines, 256)

    logging.info("Running model to compute loss")
    batch_size = batch_size
    batches = range(0, len(sft.streamlines), batch_size)

    def _autoencode_streamlines():
        for i, b in enumerate(batches):
            with torch.no_grad():
                s = np.asarray(
                    set_number_of_points(
                        bundle[i * batch_size:(i+1) * batch_size],
                        256))
                if normalize:
                    s /= sft.dimensions

                streamlines = torch.as_tensor(
                    s, dtype=torch.float32, device=device)
                tmp_outputs = model(streamlines).cpu().numpy()
                # latent = model.encode(streamlines)
                scaling = sft.dimensions if normalize else 1.0
                streamlines_output = tmp_outputs.transpose((0, 2, 1)) * scaling
                for strml in streamlines_output:
                    yield strml, strml[0]

    return _autoencode_streamlines


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    tracts_format = detect_format(args.out_tractogram)
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
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Load model
    logging.debug("Loading model.")
    if args.model == 'finta':
        architecture = ModelAE
    else:
        architecture = ModelConvNextAE

    model = architecture.load_model_from_params_and_state(
            args.experiment_path + '/best_model', log_level=sub_loggers_level
    ).to(device)

    sft = load_tractogram_with_reference(p, args, args.in_tractogram)

    _autoencode_streamlines = autoencode_streamlines(
        model, sft, device, args.batch_size, args.normalize)

    # Need a nifti image to lazy-save a tractogram
    fake_ref = nib.Nifti1Image(np.zeros(sft.dimensions), sft.affine)

    save_tractogram(_autoencode_streamlines(), tracts_format, fake_ref,
                    len(sft.streamlines), args.out_tractogram,
                    0, 999, False, False, args.verbose)


if __name__ == '__main__':
    main()
