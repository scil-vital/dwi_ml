#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script creates a connectivity matrix (using streamline count, or binary)
when you don't have labels for your data, using a division of the volume into
N blocs. Useful for supervised machine learning models.

If you do have labels, see
>> scil_connectivity_compute_simple_matrix.py

"""
import argparse
import logging
import os.path

import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
from dipy.io.utils import is_header_compatible
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_verbose_arg, add_overwrite_arg

from dwi_ml.data.hdf5.utils import format_nb_blocs_connectivity
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_triu_connectivity_from_blocs


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_volume',
                   help='Input nifti volume. Only used to get the shape of '
                        'the volume.')
    p.add_argument('streamlines',
                   help='Tractogram (trk or tck).')
    p.add_argument('out_file',
                   help="Out .npy file. Will also save it as a .png image.")
    p.add_argument(
        'connectivity_nb_blocs', metavar='m', type=int, nargs='+',
        help="Number of 3D blocks (m x m x m) for the connectivity matrix. \n"
             "(The matrix will be m^3 x m^3). If more than one values are "
             "provided, expected to be one per dimension. \n"
             "Default: 20x20x20.")
    p.add_argument('--binary', action='store_true',
                   help="If set, saves the result as binary. Else, the "
                        "streamline count is saved.")
    p.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrix with matplotlib.")

    add_verbose_arg(p)
    add_overwrite_arg(p)

    return p


def prepare_figure_connectivity(matrix):
    # Equivalent to the figure in scil_connectivity_compute_simple_matrix.py
    matrix = np.copy(matrix)

    fig, axs = plt.subplots(2, 2)
    im = axs[0, 0].imshow(matrix)
    divider = make_axes_locatable(axs[0, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 0].set_title("Raw streamline count")

    im = axs[0, 1].imshow(matrix + np.min(matrix[matrix > 0]), norm=LogNorm())
    divider = make_axes_locatable(axs[0, 1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[0, 1].set_title("Raw streamline count (log view)")

    matrix = matrix / matrix.sum() * 100
    im = axs[1, 0].imshow(matrix)
    divider = make_axes_locatable(axs[1, 0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    axs[1, 0].set_title("Percentage of the total streamline count")

    matrix = matrix > 0
    axs[1, 1].imshow(matrix)
    axs[1, 1].set_title("Binary matrix: 1 if at least 1 streamline")

    plt.suptitle("Connectivity matrix: streamline count")


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    args.connectivity_nb_blocs = format_nb_blocs_connectivity(
        args.connectivity_nb_blocs)

    tmp, ext = os.path.splitext(args.out_file)
    out_fig = tmp + '.png'
    assert_inputs_exist(p, [args.in_volume, args.streamlines])
    assert_outputs_exist(p, args, [args.out_file, out_fig],
                         [args.save_biggest, args.save_smallest])

    ext = os.path.splitext(args.streamlines)[1]
    if ext == '.trk':
        args.reference = None
        if not is_header_compatible(args.streamlines, args.in_volume):
            p.error("Streamlines not compatible with chosen volume.")
    else:
        args.reference = args.in_volume
    in_sft = load_tractogram_with_reference(p, args, args.streamlines)
    in_sft.to_vox()
    in_sft.to_corner()
    in_img = nib.load(args.in_volume)

    matrix, start_blocs, end_blocs = compute_triu_connectivity_from_blocs(
        in_sft.streamlines, in_img.shape, args.connectivity_nb_blocs)

    prepare_figure_connectivity(matrix)

    if args.binary:
        matrix = matrix > 0

    # Save results.
    np.save(args.out_file, matrix)
    plt.savefig(out_fig)

    if args.show_now:
        plt.show()


if __name__ == '__main__':
    main()
