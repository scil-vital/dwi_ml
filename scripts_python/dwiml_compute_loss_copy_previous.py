#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One difficulty of choosing a good loss function for tractography is that
streamlines have the particularity of being smooth.

Printing the average loss function for a given dataset when we simply copy the
previous direction.

    Target :=  SFT.streamlines' directions[1:]
    Y := Previous directions.
    loss = DirectionGetter(Target, Y)
"""
import argparse
import logging

import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional

from dwi_ml.io_utils import add_resample_or_compress_arg
from dwi_ml.models.projects.copy_previous_dirs import CopyPrevDirModel
from dwi_ml.models.utils.direction_getters import add_direction_getter_args, \
    check_args_direction_getter
from dwi_ml.testing.testers import Tester, load_sft_from_hdf5
from dwi_ml.testing.utils import add_args_testing_subj_hdf5
from dwi_ml.testing.visu_loss import (run_visu_save_colored_displacement,
                                      run_visu_save_colored_sft)
from dwi_ml.testing.visu_loss_utils import prepare_args_visu_loss, visu_checks

CHOICES = ['cosine-regression', 'l2-regression', 'sphere-classification',
           'smooth-sphere-classification', 'cosine-plus-l2-regression']


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_args_testing_subj_hdf5(p, ask_input_group=False,
                               ask_streamlines_group=True)
    prepare_args_visu_loss(p)
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

    # Checks
    (colored_sft_name, colorbar_name, colored_best_name,
     colored_worst_name, displacement_sft_name) = visu_checks(args, p)

    # Device
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Prepare fake model
    dg_args = check_args_direction_getter(args)
    model = CopyPrevDirModel(args.dg_key, dg_args, args.skip_first_point,
                             args.step_size, args.compress)
    model.set_context('visu')

    # 2. Load data through the tester
    tester = Tester(model, args.subj_id, args.hdf5_file, args.subset,
                    args.batch_size, device)

    # 3. Load SFT. Right now from hdf5. Could offer option to load from disk.
    sft = load_sft_from_hdf5(args.subj_id, args.hdf5_file, args.subset,
                             args.streamlines_group)

    # (Subsample if possible)
    if not (args.save_colored_tractogram or args.save_colored_best_and_worst
            or args.displacement_on_best_and_worst):
        # Only saving: displacement_on_nb.
        # Avoid running on all streamlines for no reason.
        chosen_streamlines = np.random.randint(0, len(sft),
                                               size=args.displacement_on_nb)
        sft = sft[chosen_streamlines]

    # 4. Run model
    logging.info("Running model on {} streamlines to compute loss"
                 .format(len(sft)))
    sft, outputs, losses, mean_loss_per_line = tester.run_model_on_sft(
        sft, compute_loss=True)

    if not model.direction_getter.add_eos:
        # We will not get a loss value nor a noutput for the last point of the
        # streamlines. Removing from sft.
        sft.streamlines = [line[:-1] for line in sft.streamlines]
    assert len(losses[0]) == len(sft.streamlines[0]), \
        ("Expecting one loss per point, for each streamline, but got {} for "
         "streamline 0, of len {}. Error in our code?"
         .format(len(losses[0]), len(sft.streamlines[0])))

    # 5. Colored SFT
    if args.save_colored_tractogram or args.save_colored_best_and_worst:
        run_visu_save_colored_sft(
            losses, mean_loss_per_line, model, sft,
            save_whole_tractogram=args.save_colored_tractogram,
            colored_sft_name=colored_sft_name,
            save_separate_best_and_worst=args.save_colored_best_and_worst,
            best_worst_nb=args.best_and_worst_nb,
            best_sft_name=colored_best_name, worst_sft_name=colored_worst_name,
            colorbar_name=colorbar_name, colormap=args.colormap,
            min_range=args.min_range, max_range=args.max_range)

    # 6. Displacement.
    if args.save_displacement:
        run_visu_save_colored_displacement(
            model, outputs, mean_loss_per_line, sft,
            displacement_sft_name, args.displacement_on_nb,
            args.displacement_on_best_and_worst)

    if args.show_colorbar:
        plt.show()


if __name__ == '__main__':
    main()
