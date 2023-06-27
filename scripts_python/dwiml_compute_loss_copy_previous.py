#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One difficulty of choosing a good loss function for tractography is that
streamlines have the particularity of being smooth.

Printing the average loss function for a given dataset when we simply copy the
previous direction.

    Target :=  SFT.streamlines's directions[1:]
    Y := Previous directions.
    loss = DirectionGetter(Target, Y)
"""
import argparse
import logging
import os

import torch.nn.functional

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.arg_utils import add_resample_or_compress_arg
from dwi_ml.models.projects.copy_previous_dirs import CopyPrevDirModel
from dwi_ml.models.utils.direction_getters import add_direction_getter_args, \
    check_args_direction_getter
from dwi_ml.testing.projects.copy_prev_dirs_tester import TesterCopyPrevDir
from dwi_ml.testing.visu_loss import \
    prepare_args_visu_loss, pick_a_few, run_visu_save_colored_displacement

CHOICES = ['cosine-regression', 'l2-regression', 'sphere-classification',
           'smooth-sphere-classification', 'cosine-plus-l2-regression']


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    prepare_args_visu_loss(p, use_existing_experiment=False)
    p.add_argument('streamlines_group',
                   help="Streamline group to use as SFT for the given "
                        "subject in the hdf5.")
    p.add_argument('--skip_first_point', action='store_true',
                   help="If set, do not compute the loss at the first point "
                        "of the streamline. \nElse (default) compute it with "
                        "previous dir = 0.")
    add_resample_or_compress_arg(p)

    add_direction_getter_args(p)

    return p


def main():
    p = prepare_arg_parser()
    args = p.parse_args()
    logging.getLogger().setLevel(level=args.logging)

    if args.out_displacement_sft and \
            not (args.pick_at_random or args.pick_best_and_worst or
                 args.pick_idx):
        p.error("You must select at least one of 'pick_at_random', "
                "'pick_best_and_worst' and 'pick_idx'.")

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

    # 1. Prepare fake model
    dg_args = check_args_direction_getter(args)
    model = CopyPrevDirModel(args.dg_key, dg_args, args.skip_first_point,
                             args.step_size, args.compress)

    # 2. Compute loss
    tester = TesterCopyPrevDir(model, args.streamlines_group,
                               args.batch_size, device)
    sft = tester.load_and_format_data(args.subj_id, args.hdf5_file,
                                      args.subset)

    if (args.out_displacement_sft and not args.out_colored_sft and
            not args.pick_best_and_worst):
        # Picking a few streamlines now to avoid running model on all
        # streamlines for no reason.
        sft, _ = pick_a_few(sft, [], [], args.pick_at_random,
                            args.pick_best_and_worst, args.pick_idx)

    logging.info("Running model to compute loss")
    outputs, losses = tester.run_model_on_sft(sft)

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
