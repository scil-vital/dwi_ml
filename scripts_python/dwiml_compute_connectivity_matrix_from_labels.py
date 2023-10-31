#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from dipy.io.streamline import save_tractogram
from dipy.io.utils import is_header_compatible
from scilpy.image.labels import get_data_as_labels

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    load_tractogram_with_reference, add_verbose_arg, add_overwrite_arg

from dwi_ml.data.processing.streamlines.post_processing import \
    find_streamlines_with_chosen_connectivity, \
    compute_triu_connectivity_from_labels


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
    p.add_argument('--hide_background', nargs='?', const=0,
                   help="If true, set the connectivity matrix for chosen label "
                        "(default: 0), to 0.")
    p.add_argument(
        '--use_longest_segment', action='store_true',
        help="If true, uses scilpy's method:\n"
             "  'Strategy is to keep the longest streamline segment \n"
             "   connecting 2 regions. If the streamline crosses other gray \n"
             "   matter regions before reaching its final connected region, \n"
             "   the kept connection is still the longest. This is robust to \n"
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

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    tmp, ext = os.path.splitext(args.out_file)
    out_fig = tmp + '.png'
    assert_inputs_exist(p, [args.in_labels, args.streamlines])
    assert_outputs_exist(p, args, [args.out_file, out_fig],
                         [args.save_biggest, args.save_smallest])

    ext = os.path.splitext(args.streamlines)[1]
    if ext == '.trk':
        args.reference = None
        if not is_header_compatible(args.streamlines, args.in_labels):
            p.error("Streamlines not compatible with chosen volume.")
    else:
        args.reference = args.in_labels
    in_sft = load_tractogram_with_reference(p, args, args.streamlines)
    in_img = nib.load(args.in_labels)
    data_labels = get_data_as_labels(in_img)

    in_sft.to_vox()
    in_sft.to_corner()
    matrix, start_blocs, end_blocs = compute_triu_connectivity_from_labels(
        in_sft._streamlines_getter, data_labels, use_scilpy=args.use_longest_segment)

    if args.hide_background is not None:
        matrix[args.hide_background, :] = 0
        matrix[:, args.hide_background] = 0

    # Options to try to investigate the connectivity matrix:
    # masking point (0,0) = streamline ending in wm.
    if args.save_biggest is not None:
        i, j = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)
        print("Saving biggest bundle: {} streamlines. From label {} to label "
              "{}".format(matrix[i, j], i, j))
        biggest = find_streamlines_with_chosen_connectivity(
            in_sft._streamlines_getter, i, j, start_blocs, end_blocs)
        sft = in_sft.from_sft(biggest, in_sft)
        save_tractogram(sft, args.save_biggest)

    if args.save_smallest is not None:
        tmp_matrix = np.ma.masked_equal(matrix, 0)
        i, j = np.unravel_index(tmp_matrix.argmin(axis=None), matrix.shape)
        print("Saving smallest bundle: {} streamlines. From label {} to label "
              "{}".format(matrix[i, j], i, j))
        biggest = find_streamlines_with_chosen_connectivity(
            in_sft._streamlines_getter, i, j, start_blocs, end_blocs)
        sft = in_sft.from_sft(biggest, in_sft)
        save_tractogram(sft, args.save_smallest)

    if args.show_now:
        plt.imshow(matrix)
        plt.colorbar()

        plt.figure()
        plt.imshow(matrix > 0)
        plt.title('Binary')

    if args.binary:
        matrix = matrix > 0

    # Save results.
    np.save(args.out_file, matrix)
    plt.savefig(out_fig)
    plt.show()


if __name__ == '__main__':
    main()



