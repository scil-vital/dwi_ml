#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist
from torch.nn.utils.rnn import unpack_sequence

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.unit_tests.visual_tests.visualise_error_sft import \
    build_argparser_visu_error, \
    prepare_batch_visu_error, save_output_with_ref


def main():
    p = build_argparser_visu_error()
    args = p.parse_args()

    # Prepare stuff
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        # make them info max
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    assert_inputs_exist(p, args.input_sft, args.reference)
    assert_outputs_exist(p, args, args.out_sft)

    # Load model
    logging.info("Loading model.")
    model = Learn2TrackModel.load_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)

    # Prepare batch
    (sft, batch_streamlines,
     batch_input) = prepare_batch_visu_error(p, args, model)

    # Run model
    model.eval()
    grad_context = torch.no_grad()
    with grad_context:
        outputs = model(batch_input, batch_streamlines, return_state=False)

    # Split outputs
    outputs = unpack_sequence(outputs)
    out_dirs = [model.get_tracking_directions(s_output, algo='det')
                for s_output in outputs]

    # Save error together with ref
    save_output_with_ref(out_dirs, args, sft)


if __name__ == '__main__':
    main()
