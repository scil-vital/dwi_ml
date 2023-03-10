#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.models.main_models import MainModelOneInput
from dwi_ml.tests.visual_tests.visualise_error_sft import \
    build_argparser_visu_error, \
    prepare_batch_visu_error, save_output_with_ref


def main():
    p = build_argparser_visu_error(skip_exp=True)
    p.add_argument('input_group', help="Input group in the hdf5 file.")
    args = p.parse_args()

    # Prepare stuff
    logging.getLogger().setLevel(level=args.logging)

    assert_inputs_exist(p, args.input_sft, args.reference)
    assert_outputs_exist(p, args, args.out_sft)

    # Fake model
    model = MainModelOneInput(experiment_name='TEST')

    # Prepare batch
    (sft, batch_streamlines,
     batch_input) = prepare_batch_visu_error(p, args, model,
                                             check_data_loader=False)

    directions = compute_directions(batch_streamlines)
    out_dirs = [d[:-1, :] for d in directions]
    zeros = torch.as_tensor([0, 0, 0])
    out_dirs = [torch.vstack((zeros, d)) for d in out_dirs]

    # Save error together with ref
    save_output_with_ref(out_dirs, args, sft)


if __name__ == '__main__':
    main()
