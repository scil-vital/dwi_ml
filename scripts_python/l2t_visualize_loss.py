#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os.path

import torch

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.testing.testers import TesterOneInput
from dwi_ml.testing.visu_loss import \
    prepare_args_visu_loss, pick_a_few, run_visu_save_colored_displacement


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    prepare_args_visu_loss(p)
    args = p.parse_args()

    if args.out_displacement_sft and \
            not (args.pick_at_random or args.pick_best_and_worst or
                 args.pick_idx):
        p.error("You must select at least one of 'pick_at_random', "
                "'pick_best_and_worst' and 'pick_idx'.")

    # Loggers
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    # Verify output names
    out_files = [args.out_colored_sft]
    colorbar_name, best_sft_name, worst_sft_name = (None, None, None)
    if args.out_colored_sft is not None:
        base_name, _ = os.path.splitext(os.path.basename(args.out_colored_sft))
        file_dir = os.path.dirname(args.out_colored_sft)

        colorbar_name = (os.path.join(file_dir, base_name + '_colorbar.png')
                         if args.out_colored_sft is not None else None)
        best_sft_name = (os.path.join(file_dir, base_name + '_best.trk')
                         if args.save_best_and_worst is not None else None)
        worst_sft_name = (os.path.join(file_dir, base_name + '_worst.trk')
                          if args.save_best_and_worst is not None else None)
        out_files += [colorbar_name, best_sft_name, worst_sft_name]

    assert_inputs_exist(p, args.hdf5_file)
    assert_outputs_exist(p, args, [], out_files)

    # Device
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Load model
    logging.debug("Loading model.")
    if args.use_latest_epoch:
        model = Learn2TrackModel.load_model_from_params_and_state(
            args.experiment_path + '/checkpoint/model',
            log_level=sub_logger_level)
    else:
        model = Learn2TrackModel.load_model_from_params_and_state(
            args.experiment_path + '/best_model',
            log_level=sub_logger_level)

    # 2. Compute loss
    tester = TesterOneInput(args.experiment_path, model, args.batch_size, device)
    sft = tester.load_and_format_data(args.subj_id, args.hdf5_file, args.subset)

    if (args.out_displacement_sft and not args.out_colored_sft and
            not args.pick_best_and_worst):
        # Picking a few streamlines now to avoid running model on all
        # streamlines for no reason.
        sft, _ = pick_a_few(sft, [], [], args.pick_at_random,
                            args.pick_best_and_worst, args.pick_idx)

    logging.info("Running model to compute loss")

    outputs, losses = tester.run_model_on_sft(
        sft, uncompress_loss=args.uncompress_loss,
        force_compress_loss=args.force_compress_loss,
        weight_with_angle=args.weight_with_angle)

    compute_loss_only = (args.out_colored_sft is None and
                         args.out_displacement_sft is None and
                         args.save_best_and_worst is None)
    if compute_loss_only:
        return

    run_visu_save_colored_displacement(args, model, losses, outputs, sft,
                                       colorbar_name, best_sft_name,
                                       worst_sft_name)


if __name__ == '__main__':
    main()
