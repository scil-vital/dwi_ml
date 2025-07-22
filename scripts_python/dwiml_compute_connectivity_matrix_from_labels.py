#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Computes the connectivity matrix.
Labels associated with each line / row will be printed.
"""

import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
from scilpy.image.labels import get_data_as_labels

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_verbose_arg, add_overwrite_arg

from dwi_ml.data.processing.streamlines.post_processing import \
    find_streamlines_with_chosen_connectivity, \
    compute_triu_connectivity_from_labels, prepare_figure_connectivity


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_labels',
                   help='Input nifti volume.')
    p.add_argument('streamlines',
                   help='Tractogram (trk or tck).')
    p.add_argument('out_file',
                   help="Out .npy file.")
    p.add_argument('--binary', action='store_true',
                   help="If set, saves the result as binary. Else, the "
                        "streamline count is saved.")
    p.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrix with matplotlib.")
    p.add_argument('--hide_background', nargs='?', const=0, type=int,
                   help="If true, set the connectivity matrix for chosen "
                        "label (default: 0), to 0.")
    p.add_argument(
        '--use_longest_segment', action='store_true',
        help="If true, uses scilpy's method:\n"
             "  'Strategy is to keep the longest streamline segment \n"
             "   connecting 2 regions. If the streamline crosses other gray \n"
             "   matter regions before reaching its final connected region, \n"
             "   the kept connection is still the longest. This is robust to\n"
             "   compressed streamlines.'\n"
             "Else, uses simple computation from endpoints. Faster. Also, "
             "works with incomplete parcellation.")

    g = p.add_argument_group("Investigation of the matrix:")
    g.add_argument('--save_biggest', metavar='filename',
                   help="If set, saves the biggest bundle (as tck or trk).")
    g.add_argument('--save_smallest', metavar='filename',
                   help="If set, saves the smallest (non-zero) bundle "
                        "(as tck or trk).")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def main():
    p = _build_arg_parser()
    args = p.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    # Currently, with debug, matplotlib prints a lot of stuff. Why??
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.ticker').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.colorbar').setLevel(logging.ERROR)
    logging.getLogger('matplotlib.pyplot').setLevel(logging.ERROR)
    logging.getLogger('PIL.PngImagePlugin').setLevel(logging.ERROR)

    tmp, ext = os.path.splitext(args.out_file)

    if ext != '.npy':
        p.error("--out_file should have a .npy extension.")

    out_fig = tmp + '.png'
    out_ordered_labels = tmp + '_labels.txt'
    out_rejected_streamlines = tmp + '_rejected_from_background.trk'
    assert_inputs_exist(p, [args.in_labels, args.streamlines])
    assert_outputs_exist(p, args,
                         [args.out_file, out_fig, out_rejected_streamlines],
                         [args.save_biggest, args.save_smallest])

    ext = os.path.splitext(args.streamlines)[1]
    if ext == '.trk':
        args.reference = None
        if not is_header_compatible(args.streamlines, args.in_labels):
            p.error("Streamlines not compatible with chosen volume.")
    else:
        args.reference = args.in_labels

    logging.info("Loading tractogram and labels.")
    in_sft = load_tractogram_with_reference(p, args, args.streamlines)
    in_img = nib.load(args.in_labels)
    data_labels = get_data_as_labels(in_img)

    logging.info("Tractogram contains {} streamlines."
                 .format(len(in_sft.streamlines)))

    in_sft.to_vox()
    in_sft.to_corner()
    matrix, ordered_labels, start_labels, end_labels = \
        compute_triu_connectivity_from_labels(
            in_sft.streamlines, data_labels,
            use_scilpy=args.use_longest_segment)

    if args.hide_background is not None:
        idx = ordered_labels.index(args.hide_background)
        nb_hidden = np.sum(matrix[idx, :]) + np.sum(matrix[:, idx]) - \
            matrix[idx, idx]
        if nb_hidden > 0:
            logging.warning("CAREFUL! Deleting from the matrix {} streamlines "
                            "with one or both endpoints in a non-labelled "
                            "area (background = {}; line/column {})"
                            .format(nb_hidden, args.hide_background, idx))
            rejected = find_streamlines_with_chosen_connectivity(
                in_sft.streamlines, start_labels, end_labels, idx)
            logging.info("Saving rejected streamlines in {}"
                         .format(out_rejected_streamlines))
            sft = in_sft.from_sft(rejected, in_sft)
            save_tractogram(sft, out_rejected_streamlines)
        else:
            logging.info("No streamlines with endpoints in the background :)")
        matrix[idx, :] = 0
        matrix[:, idx] = 0
        ordered_labels[idx] = ("Hidden background ({})"
                               .format(args.hide_background))

    # Save figure will all versions of the matrix.
    prepare_figure_connectivity(matrix)
    plt.savefig(out_fig)

    if args.binary:
        matrix = matrix > 0

    # Save results.
    np.save(args.out_file, matrix)

    # Options to try to investigate the connectivity matrix:
    # masking point (0,0) = streamline ending in wm.
    if args.save_biggest is not None:
        i, j = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        print("Saving biggest bundle: {} streamlines. From label {} to label "
              "{} (line {}, column {} in the matrix)"
              .format(matrix[i, j], ordered_labels[i], ordered_labels[j],
                      i, j))
        biggest = find_streamlines_with_chosen_connectivity(
            in_sft.streamlines, i, j, start_labels, end_labels)
        sft = in_sft.from_sft(biggest, in_sft)
        save_tractogram(sft, args.save_biggest)

    if args.save_smallest is not None:
        tmp_matrix = np.ma.masked_equal(matrix, 0)
        i, j = np.unravel_index(tmp_matrix.argmin(axis=None), matrix.shape)
        print("Saving smallest bundle: {} streamlines. From label {} to label "
              "{} (line {}, column {} in the matrix)"
              .format(matrix[i, j], ordered_labels[i], ordered_labels[j],
                      i, j))
        smallest = find_streamlines_with_chosen_connectivity(
            in_sft.streamlines, i, j, start_labels, end_labels)
        sft = in_sft.from_sft(smallest, in_sft)
        save_tractogram(sft, args.save_smallest)

    with open(out_ordered_labels, "w") as text_file:
        logging.info("Labels are saved in: {}".format(out_ordered_labels))
        for i, label in enumerate(ordered_labels):
            text_file.write("{} = {}\n".format(i, label))

    if args.show_now:
        plt.show()


if __name__ == '__main__':
    main()
