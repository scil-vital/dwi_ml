#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to binarize a 3D or 4D Nifti volume."""
import argparse
import os
import pathlib

import nibabel as nib
import numpy as np


def parse_args():
    """Argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('input', type=str, help="Path to input volume.")
    parser.add_argument('-t', '--threshold', type=float, default=0.,
                        help="Values under the threshold will be set to 0, "
                             "while values over the threshold will be set to 1.")
    parser.add_argument('-o', '--output', type=str, default="output.nii.gz",
                        help="Output file. Default")
    parser.add_argument('-f', '--force', action="store_true",
                        help="Force overwriting of output.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    input_file = pathlib.Path(args.input)
    output_file = pathlib.Path(args.output)
    if not input_file.exists():
        raise ValueError("Input does not exist: {}".format(str(input_file)))
    if output_file.exists():
        if not args.force:
            raise ValueError("Output file exists: {}; use --force to overwrite".format(str(output_file)))
        else:
            os.remove(str(output_file))

    input_img = nib.load(str(input_file))
    input_data = input_img.get_fdata()
    output_data = np.zeros_like(input_data)
    output_data[input_data > args.threshold] = 1
    output_img = nib.Nifti1Image(output_data, input_img.affine, input_img.header)
    nib.save(output_img, str(output_file))


if __name__ == '__main__':
    main()
