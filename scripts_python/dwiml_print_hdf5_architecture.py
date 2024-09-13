# -*- coding: utf-8 -*-
import argparse

import h5py
from scilpy.io.utils import assert_inputs_exist


def _prepare_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('hdf5_file',
                   help="Path to the hdf5 file.")
    return p


def main():
    p = _prepare_argparser()
    args = p.parse_args()

    assert_inputs_exist(p, args.hdf5_file)

    with h5py.File(args.hdf5_file, 'r') as hdf_handle:

        print("\n\nHere is the architecture of your hdf5:\n"
              "--------------------------------------\n")
        print("- Main hdf5 attributes: {}\n"
              .format(list(hdf_handle.attrs.keys())))

        if 'training_subjs' in hdf_handle.attrs:
            print("- List of training subjects: {}\n"
                  .format(hdf_handle.attrs['training_subjs']))

        if 'validation_subjs' in hdf_handle.attrs:
            print("- List of validation subjects: {}\n"
                  .format(hdf_handle.attrs['validation_subjs']))

        if 'testing_subjs' in hdf_handle.attrs:
            print("- List of testing subjects: {}\n"
                  .format(hdf_handle.attrs['testing_subjs']))

        print("- For each subject, caracteristics are:")
        first_subj = list(hdf_handle.keys())[0]
        for key, val in hdf_handle[first_subj].items():
            print("   - {}, with attributes {}"
                  .format(key, list(hdf_handle[first_subj][key].attrs.keys())))


if __name__ == '__main__':
    main()
