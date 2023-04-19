#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os.path

from dipy.io.streamline import save_tractogram
import matplotlib.pyplot as plt
import torch

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist

from dwi_ml.models.projects.transforming_tractography import \
    TransformerSrcAndTgtModel
from dwi_ml.testing.testers import TesterOneInput

from dwi_ml.visu.visu_loss import prepare_colors_from_loss, \
    prepare_args_visu_loss, combine_displacement_with_ref, pick_a_few, \
    separate_best_and_worst


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    prepare_args_visu_loss(p)
    args = p.parse_args()

    if not (args.pick_at_random or args.pick_best_and_worst or
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
    model = TransformerSrcAndTgtModel.load_params_and_state(
        args.experiment_path + '/best_model', log_level=sub_logger_level)

    # 2. Compute loss
    tester = TesterOneInput(args.experiment_path, model, args.batch_size,
                            device)
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

    # 3. Save colored SFT
    if args.out_colored_sft is not None:
        logging.info("Preparing colored sft")
        sft, colorbar_fig = prepare_colors_from_loss(
            losses, model.direction_getter.add_eos, sft,
            args.colormap, args.min_range, args.max_range)
        print("Saving colored SFT as {}".format(args.out_colored_sft))
        save_tractogram(sft, args.out_colored_sft)

        print("Saving colorbar as {}".format(colorbar_name))
        colorbar_fig.savefig(colorbar_name)

    # 4. Separate best and worst
    best_idx = []
    worst_idx = []
    if args.save_best_and_worst is not None or args.pick_best_and_worst:
        best_idx, worst_idx = separate_best_and_worst(
            args.save_best_and_worst, model.direction_getter.add_eos,
            losses, sft)

        if args.out_colored is not None:
            best_sft = sft[best_idx]
            worst_sft = sft[worst_idx]
            print("Saving best and worst streamlines as {} \nand {}"
                  .format(best_sft_name, worst_sft_name))
            save_tractogram(best_sft, best_sft_name)
            save_tractogram(worst_sft, worst_sft_name)

    # 4. Save displacement
    if args.out_displacement_sft:
        if model.direction_getter.add_eos:
            outputs = torch.split(outputs, [len(s) for s in sft.streamlines])
        else:
            outputs = torch.split(outputs, [len(s) - 1 for s in sft.streamlines])

        if args.out_colored_sft:
            # We have run model on all streamlines. Picking a few now.
            sft, idx = pick_a_few(
                sft, best_idx, worst_idx, args.pick_at_random,
                args.pick_best_and_worst, args.pick_idx)
            outputs = [outputs[i] for i in idx]

        # Either concat, run, split or:
        out_dirs = [model.get_tracking_directions(s_output, algo='det').numpy()
                    for s_output in outputs]

        # Save error together with ref
        sft = combine_displacement_with_ref(out_dirs, sft, model.step_size)

        save_tractogram(sft, args.out_displacement_sft, bbox_valid_check=False)

    if args.show_colorbar:
        plt.show()


if __name__ == '__main__':
    main()
