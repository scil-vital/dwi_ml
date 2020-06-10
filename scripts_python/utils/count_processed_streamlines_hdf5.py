#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to count the number of streamlines in a processed dataset after
subsampling. It accepts an arbitrary number of .hdf5 as input.;
"""

import sys

import h5py


def main():
    files = sys.argv[1:]
    for filepath in files:
        with h5py.File(filepath, 'r') as hdf_file:
            subjects = list(hdf_file.keys())
            total_streamlines = 0
            for subject_id in subjects:
                streamlines_group = hdf_file["{}/streamlines".
                    format(subject_id)]
                subject_streamlines = len(streamlines_group['offsets'])
                total_streamlines += subject_streamlines

        print("Total number of streamlines: {} in file {}".
              format(total_streamlines, filepath))


if __name__ == '__main__':
    main()
