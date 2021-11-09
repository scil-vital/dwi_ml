#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to orient streamlines according to a bundle head mask.
Given a tractogram and a bundle head mask, this script will 1) remove streamlines
without an endpoint in the mask, and 2) flip streamlines in order for the first
point of each streamline to be in the mask."""
import argparse
import os
from argparse import RawTextHelpFormatter
from typing import Tuple

import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram


def _get_voxel(idx: np.ndarray) -> np.ndarray:
    return np.floor(idx)


def _is_coords_valid(coord: np.ndarray, valid_voxels: Tuple[np.ndarray, np.ndarray, np.ndarray]):
    vox = _get_voxel(coord)
    for i in np.where(valid_voxels[0] == vox[0])[0]:
        if valid_voxels[1][i] == vox[1] and valid_voxels[2][i] == vox[2]:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('input', help="Input tractogram (.tck|.trk)")
    parser.add_argument('ref', help="Reference to load the tractogram (.nii.gz)")
    parser.add_argument('mask', help="Bundle head mask (.nii.gz)")
    parser.add_argument('output', help="Output tractogram (.tck|.trk)")
    parser.add_argument('-f', '--force', action="store_true",
                        help="Overwrite output if it exists.")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.output) and not args.force:
        raise ValueError("Output already exists! Use --force to overwrite.")

    sft = load_tractogram(args.input, args.ref,
                          to_space=Space.RASMM,
                          trk_header_check=False,
                          bbox_valid_check=False)

    # There may be invalid streamlines in the input tractogram
    sft.remove_invalid_streamlines()

    # Work in voxel space, and move streamlines to corner so we can use floor()
    # to compare with valid voxel coordinates.
    sft.to_vox()
    sft.to_corner()

    mask = nib.load(args.mask)

    valid_voxels = np.where(mask.get_fdata() > 0.5)

    valid_streamlines = []

    for s in sft.streamlines:
        if _is_coords_valid(s[0], valid_voxels):
            valid_streamlines.append(s)
        elif _is_coords_valid(s[-1], valid_voxels):
            valid_streamlines.append(s[::-1])

    valid_sft = StatefulTractogram(valid_streamlines, args.ref, space=sft.space, shifted_origin=sft.shifted_origin)
    save_tractogram(valid_sft, args.output)


if __name__ == '__main__':
    main()
