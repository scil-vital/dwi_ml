#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import nibabel as nb
from nibabel.streamlines import detect_format
import numpy as np

import torch

from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram
from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                             add_memory_args)
from dwi_ml.data.processing.streamlines.utils import _autoencode_streamlines
from dwi_ml.models.projects.ae_models import ModelAE


def _build_arg_parser():
    p = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                description=__doc__)

    add_arg_existing_experiment_path(p)
    p.add_argument('in_tractogram',
                   help="If set, saves the tractogram with the loss per point "
                        "as a data per point (color)")

    p.add_argument('out_tractogram',
                   help="If set, saves the tractogram with the loss per point "
                        "as a data per point (color)")

    # Options
    p.add_argument('--batch_size', type=int, default=5000)
    p.add_argument('--normalize', action='store_true',
                   help="If set, normalize the input data "
                        "before running the model.")
    add_memory_args(p)
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
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Load model
    logging.debug("Loading model.")
    model = ModelAE.load_model_from_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_loggers_level)

    # 2. Load tractogram
    sft = load_tractogram_with_reference(p, args, args.in_tractogram)
    tracts_format = detect_format(args.out_tractogram)
    sft.to_vox()
    sft.to_corner()

    logging.info("Running Model")
    # Need a nifti image to lazy-save a tractogram
    fake_ref = nb.Nifti1Image(np.zeros(sft.dimensions), sft.affine)

    save_tractogram(_autoencode_streamlines(model,
                                            sft,
                                            args.batch_size,
                                            sft,
                                            device),
                    tracts_format=tracts_format,
                    ref_img=fake_ref,
                    total_nb_seeds=len(sft.streamlines),
                    out_tractogram=args.out_tractogram,
                    min_length=0,
                    max_length=999,
                    compress=False,
                    save_seeds=False,
                    verbose=args.verbose)


if __name__ == '__main__':
    main()
