#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
List of steps dwi_ml users could possibly want to do as preprocessing and that
are not yet available in tractoflow. To use this, we suppose you have already
ran tractoflow and have kept all the results.

We want to make sure that all the preprocessing is ran beforehand, so that we
you create your hdf5 file, nearly no heavy computing is left.
"""

from dwi_ml.data.processing.dwi.dwi import resample_raw_dwi_from_sh


def main():

    raise NotImplementedError


if __name__ == '__main__':
    main()
