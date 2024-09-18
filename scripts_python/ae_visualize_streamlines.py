#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import torch

from scilpy.io.utils import (add_overwrite_arg,
                             assert_outputs_exist,
                             add_reference_arg,
                             add_verbose_arg)
from scilpy.io.streamlines import load_tractogram_with_reference
from dipy.io.streamline import save_tractogram
from dwi_ml.io_utils import (add_arg_existing_experiment_path,
                           add_memory_args)
from dwi_ml.models.projects.ae_models import ModelAE


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

    # Loggers
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

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
        args.experiment_path + '/best_model', log_level=sub_logger_level)
    # model.set_context('training')
    # 2. Compute loss
    # tester = TesterOneInput(args.experiment_path,
    #                         model,
    #                         args.batch_size,
    #                         device)
    # tester = Tester(args.experiment_path, model, args.batch_size, device)
    # sft = tester.load_and_format_data(args.subj_id,
    #                                   args.hdf5_file,
    #                                   args.subset)

    sft = load_tractogram_with_reference(p, args, args.in_tractogram)
    sft.to_vox()
    sft.to_corner()
    bundle = sft.streamlines[0:5000]

    logging.info("Running model to compute loss")

    new_sft = sft.from_sft(bundle, sft)
    save_tractogram(new_sft, 'orig_5000.trk')

    with torch.no_grad():
        streamlines = [
            torch.as_tensor(s, dtype=torch.float32, device=device)
            for s in bundle]
        tmp_outputs = model(streamlines)
        # latent = model.encode(streamlines)

    streamlines_output = [tmp_outputs[i, :, :].transpose(0, 1).cpu().numpy() for i in range(len(bundle))]

    # print(streamlines_output[0].shape)
    new_sft = sft.from_sft(streamlines_output, sft)
    save_tractogram(new_sft, args.out_tractogram)

    # latent_output = [s.cpu().numpy() for s in latent]

    # outputs, losses = tester.run_model_on_sft(
    #    sft, uncompress_loss=args.uncompress_loss,
    #    force_compress_loss=args.force_compress_loss,
    #    weight_with_angle=args.weight_with_angle)


if __name__ == '__main__':
    main()
