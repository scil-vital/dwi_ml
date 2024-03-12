#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np

from scilpy.io.utils import assert_inputs_exist, add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('ref_matrix',
                   help="Matrix file (reference). .npy")
    p.add_argument('scored_matrix',
                   help="Matrix being scored. .npy")
    p.add_argument('--show_now', action='store_true',
                   help="If set, shows the matrix with matplotlib.")
    add_overwrite_arg(p)

    return p


def main():
    p = _build_arg_parser()
    args = p.parse_args()

    assert_inputs_exist(p, [args.ref_matrix, args.scored_matrix])

    ref = np.load(args.ref_matrix)
    m = np.load(args.scored_matrix)

    # Both the same type?
    if np.max(ref) == 1:
        bin_ref = True
        print("Reference matrix is binary.")
    else:
        bin_ref = False
        print("Sum (i.e. number of streamlines) in ref matrix: {}"
              .format(np.sum(ref)))

    if np.max(m) == 1:
        bin_m = True
        print("WARNING. The scored matrix is binary. The final score will "
              "not be a good representation of the result.")
        print("Scored matrix is binary.")
    else:
        bin_m = False
        print("Sum (i.e. number of streamlines) in scored matrix: {}"
              .format(np.sum(ref)))

    m = m.astype(int)
    ref = ref.astype(int)

    if not (bin_m or bin_ref):
        # Figure 1: Raw matrices
        fig, axs = plt.subplots(1, 3)
        im = axs[0].imshow(ref)
        fig.colorbar(im, ax=axs[0])
        im = axs[1].imshow(m)
        fig.colorbar(im, ax=axs[1])
        im = axs[2].imshow((ref - m)**2)
        fig.colorbar(im, ax=axs[2])
        plt.suptitle("Raw")

        rms = np.average((m - ref) ** 2)
        print("Root mean square of raw matrices: {}".format(rms))

        tmp_m = m / m.sum()
        tmp_ref = ref / ref.sum()
        rms = np.average((tmp_m - tmp_ref) ** 2)
        print("Root mean square of normalized matrices: {}".format(rms))
    else:
        print("At least one matrix is binary. Not computing the root mean "
              "square.")

    # Ensuring ref is binary
    ref = ref > 0

    nb_lines = m.sum()
    where_one = np.where(m > 0)
    score = np.sum(m[where_one] * (1.0 - ref[where_one])) / nb_lines
    print("Matrix's connectivity loss (corresponds to the percentage of "
          "false positive) is: {}%%".format(score * 100))

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(ref)
    axs[1].imshow(m)
    axs[2].imshow((ref.astype(int) - m.astype(int))**2)
    plt.suptitle("Binary")

    if args.show_now:
        plt.show()
    else:
        logging.warning("Saving of figure not implemented yet! Use --show_now "
                        "to see the figure.")


if __name__ == '__main__':
    main()
