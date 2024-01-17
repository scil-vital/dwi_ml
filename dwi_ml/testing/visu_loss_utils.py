# -*- coding: utf-8 -*-
import os.path
from argparse import ArgumentParser

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist)

from dwi_ml.io_utils import add_memory_args, add_logging_arg


def prepare_args_visu_loss(p: ArgumentParser):

    p.add_argument('--out_prefix', metavar='name',
                   help="Prefix for the outputs. Do not include a path. "
                        "Suffixes will be:\n"
                        "--save_colored_tractogram: _colored_all.trk\n"
                        "--save_colored_best_and_worst: "
                        "_colored_best.trk and _colored_worst.trk\n"
                        "   Any of the two predecent: _colorbar.png\n"
                        "--save_displacement: _displacement.trk")
    p.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu_loss")

    # Options
    p.add_argument('--batch_size', metavar='s', type=int,
                   help="The batch size in number of streamlines.")
    add_memory_args(p)

    g = p.add_argument_group("Options to save loss as a colored SFT")
    g.add_argument('--save_colored_tractogram', action='store_true',
                   help="If set, saves the tractogram with the loss per point "
                        "as \ndata_per_point (color).")
    g.add_argument('--save_colored_best_and_worst', type=int, const=10,
                   help="Save separately the worst x%% and best x%% of "
                        "streamlines. Default: 10%%.\n")
    g.add_argument('--colormap', default='plasma', metavar='name',
                   help="Select the colormap for colored trk [%(default)s].\n"
                        "Can be any matplotlib cmap.")
    g.add_argument('--min_range', type=float, metavar='m',
                   help="Inferior range of the colormap. If any loss is lower"
                        "than that \nvalue, they will be clipped.")
    g.add_argument('--max_range', type=float, metavar='M',
                   help='Superior range.')
    g.add_argument('--show_now', action='store_true',
                   help="If set, shows the colorbar now. Else, only saves it.")

    g = p.add_argument_group("Options to save output direction as "
                             "displacement")
    g.add_argument('--save_displacement', action='store_true',
                   help="If set, picks a few streamlines (see options below) "
                        "and computes the \noutputs at each position. Saves a "
                        "streamline that starts at each real \ncoordinate and "
                        "moves in the output direction.")
    g.add_argument('--displacement_on_nb', default=0, type=int,
                   help="Will pick n streamlines at random. Default: 0.")
    g.add_argument('--displacement_on_best_and_worst', action='store_true',
                   help="Will show displacement on best and worst streamlines "
                        "(base on loss).")

    add_overwrite_arg(p)
    add_logging_arg(p)


def visu_checks(args, parser):
    # Verifications
    if not (args.save_colored_tractogram or args.save_colored_best_and_worst
            or args.save_displacement):
        parser.error(
            "Nothing to be run. Choose at least one of "
            "--save_colored_tractogram, --save_colored_best_and_worst, or "
            "--save_displacement.")

    if args.save_displacement and not (args.displacement_on_nb or
                                       args.displacement_on_best_and_worst):
        parser.error(
            "Please tell us which displacement we should save: pick \n"
            "--displacement_on_any and/or --displacement_on_best_and_worst.")

    # Verify output names
    out_files = []
    colored_sft_name, colorbar_name = (None, None)
    colored_best_name, colored_worst_name = (None, None)
    displacement_sft_name = None

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    if args.save_colored_tractogram:
        colored_sft_name = os.path.join(
            args.out_dir, args.out_prefix + '_colored_all.trk')
        out_files += [colored_sft_name]

    if args.save_colored_best_and_worst:
        colored_best_name = os.path.join(
            args.out_dir, args.out_prefix + '_colored_best.trk')
        colored_worst_name = os.path.join(
            args.out_dir, args.out_prefix + '_colored_worst.trk')
        out_files += [colored_best_name, colored_worst_name]

    if args.save_colored_tractogram or args.save_colored_best_and_worst:
        colorbar_name = os.path.join(
            args.out_dir, args.out_prefix + '_colorbar.png')
        out_files += [colorbar_name]

    if args.save_displacement:
        displacement_sft_name = os.path.join(
            args.out_dir, args.out_prefix + '_displacement.trk')
        out_files += [displacement_sft_name]

    assert_inputs_exist(parser, args.hdf5_file)
    assert_outputs_exist(parser, args, [], out_files)

    return (colored_sft_name, colorbar_name, colored_best_name,
            colored_worst_name, displacement_sft_name)
