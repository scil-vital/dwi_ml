#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse

import nibabel as nib
import numpy as np

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_overwrite_arg


def _build_arg_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('in_hdf5',
                   help='Input hdf5 file.')
    p.add_argument('streamline_group',
                   help='Streamline group from which to compute connectivity.')
    p.add_argument('ref_volume_group',
                   help="Volume group to use as reference to divide into blocs.")
    p.add_argument(
        'nb_blocs',  metavar='m', type=int, nargs='+',
        help="Number of 3D blocks (m x m x m) for the connectivity matrix. \n"
             "(The matrix will be m^3 x m^3). If more than one values are "
             "provided, expected to be one per dimension. \n"
             "Default: 20x20x20.")
    return p

