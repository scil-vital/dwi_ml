# -*- coding: utf-8 -*-
import glob
import logging
import os.path
from argparse import ArgumentParser

from scilpy.io.utils import (add_overwrite_arg,
                             assert_inputs_exist, assert_outputs_exist,
                             add_reference_arg, ranged_type)

from dwi_ml.io_utils import add_memory_args, add_verbose_arg


def prepare_args_visu_loss(p: ArgumentParser):
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        '--streamlines_group',
        help="Streamlines group in the hdf5 to use.")
    g.add_argument(
        '--streamlines_file',
        help="Optionally, instead of using streamlines in the hdf5, you may\n"
             "provide your own streamlines to use.\n"
             "Use with care: they must correspond with the hdf5's input data.\n"
             "Offered for easier visualisation of sub-divisions of your test\n"
             "tractogram.")
    p.add_argument(
        '--out_prefix', metavar='name',
        help="Prefix for the outputs. Do not include a path. Suffixes will "
             "be:\n"
             "--show_histogram: _histogram.png\n"
             "--save_colored_tractogram: _colored.trk\n"
             "  (or _colored_clippedm-M.trk with range options)\n"
             "--save_colored_from_eos: _colored_eos.trk\n"
             "  (or _colored_eos_clippedm-M.trk with range options)\n"
             "--save_colored_best_and_worst: _colored_best.trk and "
             "_colored_worst.trk\n"
             "   Any of the three previous options: _colorbar_*.png\n"
             "--save_displacement: _displacement.trk")
    p.add_argument(
        '--out_dir', metavar='d',
        help="Output directory where to save the output files.\n"
             "Default: experiment_path/visu_loss")
    p.add_argument(
        '--show_now', action='store_true',
        help="If set, shows the matplotlib figures now (colorbar/histogram).\n"
             "Else, only saves them.")
    g.add_argument(
        "--fig_size", metavar='x y', nargs=2, default=[10, 15], type=int,
        help="Figure size (x, y) for the histogram. Default: 10x15.")

    # Options
    g = add_memory_args(p)
    g.add_argument('--batch_size', metavar='n', type=int,
                   help="The batch size in number of streamlines.")

    # --------
    g = p.add_argument_group("Options to save the histogram")
    g.add_argument('--compute_histogram', action='store_true',
                   help="If set, plots the histogram of losses (per point).")

    # --------
    g = p.add_argument_group("Options to save loss as a colored SFT")
    g.add_argument('--save_colored_tractogram', action='store_true',
                   help="If set, saves the tractogram with the loss per point "
                        "as \ndata_per_point (color).")
    g.add_argument('--save_colored_best_and_worst', metavar='percentage',
                   type=ranged_type(float, 0, 100), nargs='?', const='10',
                   help="If set, saves separately the worst x%% and best x%% "
                        "streamlines. \nOptional int to give is a number "
                        "between 0-100: the percentile of \nstreamlines "
                        "to separate as best/worst. Default: 10%%.")
    g.add_argument('--colormap', default='plasma', metavar='name',
                   help="Select the colormap for colored trk [%(default)s].\n"
                        "Can be any matplotlib cmap.")
    g.add_argument('--min_range', type=float, metavar='m',
                   help="Inferior range of the colormap. If any loss is lower"
                        "than that \nvalue, they will be clipped.")
    g.add_argument('--max_range', type=float, metavar='M',
                   help='Superior range.')

    # --------
    g = p.add_argument_group("Options to save the EOS part of the loss as a "
                             "colored SFT")
    g.add_argument('--save_colored_eos', action='store_true',
                   help="If set, saves the tractogram with the eos part of "
                        "the loss per point as \ndata_per_point (color).")
    g.add_argument('--min_range_eos', type=float, metavar='m',
                   help="Inferior range of the colormap. If any loss is lower"
                        "than that \nvalue, they will be clipped.")
    g.add_argument('--max_range_eos', type=float, metavar='M',
                   help='Superior range.')

    # --------
    g = p.add_argument_group("Options to save output direction as "
                             "displacement")
    g.add_argument(
        '--save_displacement', metavar='nb', type=int, nargs='?', const='1',
        help="If set, picks a few streamlines (given_value, default: 1) "
             "and computes the \noutputs at each position. Saves a "
             "streamline that starts at each real \ncoordinate and "
             "moves in the output direction.")

    add_overwrite_arg(p)
    add_verbose_arg(p)
    add_reference_arg(p)


def visu_checks(args, parser):
    """
    Verification of options
    +
    Preparation of all output names.
    """
    # Verifications
    if not (args.compute_histogram or args.save_colored_tractogram or
            args.save_colored_best_and_worst or args.save_displacement or
            args.save_colored_eos):
        parser.error(
            "Nothing to be run. Choose at least one of "
            "--compute_histogram, --save_colored_tractogram, "
            "--saved_colored_eos",
            "--save_colored_best_and_worst, or --save_displacement.")

    if args.min_range is not None and args.max_range is not None and \
            args.min_range >= args.max_range:
        parser.error("Min range should be smaller than max range")

    # Verify output names
    out_files = []
    histogram_name, histogram_name_eos = (None, None)
    colored_sft_name, colorbar_name = (None, None)
    colored_eos_name, colorbar_eos_name = (None, None)
    colored_best_name, colored_worst_name = (None, None)
    displacement_sft_name = None

    if not os.path.isdir(args.out_dir):
        os.mkdir(args.out_dir)

    prefix = os.path.join(args.out_dir, args.out_prefix)
    if args.compute_histogram:
        histogram_name = prefix + '_histogram.png'
        histogram_name_eos = prefix + '_histogram_eos.png'
        out_files += [histogram_name, histogram_name_eos]

    if args.save_colored_tractogram or args.save_colored_best_and_worst:
        if args.min_range is not None and args.max_range is not None:
            suffix = "_clipped{}-{}".format(args.min_range, args.max_range)
        elif args.min_range is not None:
            suffix = "_clipped{}".format(args.min_range)
        elif args.max_range is not None:
            suffix = "_clipped{}".format(args.max_range)
        else:
            suffix = ''

        if args.save_colored_tractogram:
            colored_sft_name = prefix + '_colored' + suffix + '.trk'
            out_files += [colored_sft_name]

        if args.save_colored_best_and_worst:
            p = args.save_colored_best_and_worst
            total_suffix = '{}percent'.format(p) + suffix + '.trk'
            colored_best_name = prefix + '_best' + total_suffix
            colored_worst_name = prefix + '_worst' + total_suffix
            out_files += [colored_best_name, colored_worst_name]

        colorbar_name = prefix + '_colorbar' + suffix + '.png'
        out_files += [colorbar_name]

    if args.save_colored_eos:
        if args.min_range_eos is not None and args.max_range_eos is not None:
            suffix = "_clipped{}-{}".format(args.min_range_eos,
                                            args.max_range_eos)
        elif args.min_range is not None:
            suffix = "_clipped{}".format(args.min_range_eos)
        elif args.max_range is not None:
            suffix = "_clipped{}".format(args.max_range_eos)
        else:
            suffix = ''
        colored_eos_name = prefix + '_colored_eos' + suffix + '.trk'
        colorbar_eos_name = prefix + '_colorbar_eos' + suffix + '.png'
        out_files += [colored_eos_name, colorbar_eos_name]

    if args.save_displacement:
        displacement_sft_name = prefix + '_displacement.trk'
        out_files += [displacement_sft_name]

    assert_inputs_exist(parser, args.hdf5_file,
                        [args.streamlines_file, args.reference])
    assert_outputs_exist(parser, args, [], out_files)

    untouched_files = glob.glob(prefix + '*')
    untouched_files = set(untouched_files) - set(out_files)
    if len(untouched_files) > 0:
        logging.warning("The following files, with same prefix, will NOT be "
                        "modified:\n{}".format("\n".join(untouched_files)))

    return (histogram_name, histogram_name_eos, colored_sft_name,
            colorbar_name, colored_best_name, colored_worst_name,
            colored_eos_name, colorbar_eos_name,
            displacement_sft_name)
