#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Script to extract tractograms from a generated .hdf5 dataset and save
them as .tck files"""
import argparse
import pathlib

import h5py
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import save_tractogram


def parse_args():
    """Argument parsing."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('dataset', type=str, help="Path to hdf5 dataset")
    parser.add_argument('--prefix', type=str, help="Prefix for output tractograms")
    parser.add_argument('--subject_ids', type=str, nargs='+',
                        help="List of subjects ids to extract")

    arguments = parser.parse_args()

    return arguments


def main():
    """Parse arguments, extract tractograms and save them on disk."""
    args = parse_args()

    with h5py.File(args.dataset, 'r') as hdf_file:
        if args.subject_ids:
            subject_ids = args.subject_ids
        else:
            subject_ids = list(hdf_file.keys())

        for subject_id in subject_ids:
            data = np.array(hdf_file[subject_id]['streamlines/data'])
            offsets = np.array(hdf_file[subject_id]['streamlines/offsets'])
            lengths = np.array(hdf_file[subject_id]['streamlines/lengths'])

            vox2rasmm_affine = np.array(hdf_file[subject_id]['input_volume'].attrs['vox2rasmm'])
            img = nib.Nifti1Image(np.array(hdf_file[subject_id]['input_volume/data']), vox2rasmm_affine)

            streamlines_vox = nib.streamlines.ArraySequence()
            streamlines_vox._data = data
            streamlines_vox._offsets = offsets
            streamlines_vox._lengths = lengths

            tractogram = StatefulTractogram(streamlines_vox, img,
                                            space=Space.VOX, shifted_origin=True)

            # Save tractogram
            fname = "{}_{}.tck".format(pathlib.Path(args.dataset).stem, subject_id)
            if args.prefix:
                fname = "{}_{}".format(args.prefix, fname)
            save_tractogram(tractogram, fname)


if __name__ == '__main__':
    main()
