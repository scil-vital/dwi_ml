#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt

from dwi_ml.io_utils import add_arg_existing_experiment_path
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.testing.testers import TesterOneInput, load_sft_from_hdf5
from dwi_ml.testing.utils import add_args_testing_subj_hdf5
from dwi_ml.testing.visu_loss import (run_visu_save_colored_displacement,
                                      run_visu_save_colored_sft)
from dwi_ml.testing.visu_loss_utils import prepare_args_visu_loss, visu_checks


def prepare_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_arg_existing_experiment_path(p)
    add_args_testing_subj_hdf5(p, ask_input_group=True,
                               ask_streamlines_group=True)
    prepare_args_visu_loss(p)
    return p


def main():
    p = prepare_argparser()
    args = p.parse_args()

    # Checks
    if args.out_dir is None:
        args.out_dir = os.path.join(args.experiment_path, 'visu_loss')
    (colored_sft_name, colorbar_name, colored_best_name,
     colored_worst_name, displacement_sft_name) = visu_checks(args, p)
    if not os.path.isdir(args.experiment_path):
        p.error("Experiment {} not found.".format(args.experiment_path))

    # Loggers
    sub_logger_level = args.logging.upper()
    if sub_logger_level == 'DEBUG':
        sub_logger_level = 'INFO'
    logging.getLogger().setLevel(level=args.logging)

    # Device
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Load model
    logging.debug("Loading model.")
    model = Learn2TrackModel.load_model_from_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)
    model.set_context('visu')

    # 2. Load data through the tester
    tester = TesterOneInput(
        model=model, batch_size=args.batch_size, device=device,
        subj_id=args.subj_id, hdf5_file=args.hdf5_file,
        subset_name=args.subset, volume_group=args.input_group)

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
        # We will not get a loss value nor an output for the last point of the
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

    if model.direction_getter.add_eos:
        # toDo : Save EOS prob at each point.
        print("EOS prob: toDo.")

    if args.show_colorbar:
        plt.show()


if __name__ == '__main__':
    main()
