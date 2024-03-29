#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_overwrite_arg

from dwi_ml.data.hdf5.utils import format_nb_blocs_connectivity


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_image', metavar='IN_FILE',
                   help='Input file name, in nifti format.')
    p.add_argument('out_filename',
                   help='name of the output file, which will be saved as a '
                        'text file.')
    p.add_argument('nb_blocs', nargs='+', type=int,
                   help="Number of blocs. Either a single int, or a list of "
                        "3 values.")

    add_overwrite_arg(p)

    return p


def color_mri_connectivity_blocs(nb_blocs, volume_size):

    # For tracking coordinates: we can work with float.
    # Here, dividing as ints.
    volume_size = np.asarray(volume_size)
    nb_blocs = np.asarray(nb_blocs)
    sizex, sizey, sizez = (volume_size / nb_blocs).astype(int)
    print("Coloring into blocs of size: ", sizex, sizey, sizez)

    final_volume = np.zeros(volume_size)
    for i in range(nb_blocs[0]):
        for j in range(nb_blocs[1]):
            for k in range(nb_blocs[2]):
                final_volume[i*sizex: (i+1)*sizex,
                             j*sizey: (j+1)*sizey,
                             k*sizez: (k+1)*sizez] = i + 10*j + 100*k

    return final_volume


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    # Checks
    assert_inputs_exist(parser, args.in_image)
    assert_outputs_exist(parser, args, required=args.out_filename)
    args.nb_blocs = format_nb_blocs_connectivity(args.nb_blocs)

    # Loading
    volume = nib.load(args.in_image)

    # Processing
    final_volume = color_mri_connectivity_blocs(args.nb_blocs, volume.shape)

    # Saving
    img = nib.Nifti1Image(final_volume, volume.affine)
    nib.save(img, args.out_filename)


if __name__ == '__main__':
    main()
