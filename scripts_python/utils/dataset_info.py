#!/usr/bin/env python
import argparse
import json
from argparse import RawTextHelpFormatter

import h5py


def parse_args():
    """Print dataset information"""
    parser = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument("file", help="Path to .hdf5 dataset file")

    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.file, 'r') as hdf_file:
        attrs = dict(hdf_file.attrs)
        print(json.dumps(attrs, indent=4, default=(lambda x: str(x))))


if __name__ == '__main__':
    main()
