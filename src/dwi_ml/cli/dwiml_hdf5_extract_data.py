# -*- coding: utf-8 -*-
import argparse
import logging

import h5py
import nibabel as nib
import numpy as np
from dipy.io.stateful_tractogram import StatefulTractogram, Space, Origin
from dipy.io.streamline import save_tractogram
from matplotlib import pyplot as plt

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_overwrite_arg, add_verbose_arg

from dwi_ml.data.dataset.streamline_containers import \
    load_all_streamlines_from_hdf, load_streamlines_attributes_from_hdf
from dwi_ml.data.processing.streamlines.post_processing import \
    prepare_figure_connectivity


def _prepare_argparser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('hdf5_file',
                   help="Path to the hdf5 file.")
    p.add_argument('subject',
                   help="Subject ID.")
    p.add_argument('group_id',
                   help="Name of the hdf5 group representing the data to be "
                        "retrieved.")
    p.add_argument('outname',
                   help="Name of the output file.")

    p.add_argument('--save_connectivity', action='store_true',
                   help="For a streamlines group, save the connectivity "
                        "matrix rather than the tractogram")
    add_overwrite_arg(p)
    add_verbose_arg(p)
    return p


def extract_volume(group, outname):
    if not outname.endswith('.nii.gz'):
        raise ValueError("outname must end with '.nii.gz' to save a volume.")
    data = np.asarray(group['data'])
    affine = np.asarray(group.attrs['affine'])
    img = nib.Nifti1Image(data, affine)
    logging.info("Saving volume as {}".format(outname))
    nib.save(img, outname)


def extract_sft(group, outname):
    if not (outname.endswith('.tck') or outname.endswith('.trk')):
        raise ValueError("outname must end with '.tck' or '.trk' to save a "
                         "tractogram")
    data, dps = load_all_streamlines_from_hdf(group)
    space_attrs, space, origin = load_streamlines_attributes_from_hdf(group)
    sft = StatefulTractogram(data, space_attrs, space=space, origin=origin)
    logging.info("Saving tractogram as {}".format(outname))
    save_tractogram(sft, outname)


def extract_connectivity(group, outname):
    accepted_ext= ['.jpg', '.png', '.npy']
    if not outname[-5:] not in accepted_ext:
        raise ValueError("Expecting of out the following extensions for "
                         "outname: {}".format(accepted_ext))
    matrix = np.asarray(group['connectivity_matrix'])

    if outname[-5:] == '.npy':
        np.save(outname, matrix)
    else:
        prepare_figure_connectivity(matrix)
        plt.savefig(outname)
        plt.show()

def main():
    p = _prepare_argparser()
    args = p.parse_args()
    logging.getLogger().setLevel(logging.getLevelName(args.verbose))

    assert_inputs_exist(p, args.hdf5_file)
    assert_outputs_exist(p, args, args.outname)

    with h5py.File(args.hdf5_file, 'r') as hdf_handle:
        if args.subject not in hdf_handle:
            raise ValueError("Subject {} not found.".format(args.subject))

        subj=hdf_handle[args.subject]
        if args.group_id not in subj:
            raise ValueError("Group {} not found.".format(args.group_id))

        group = subj[args.group_id]
        if 'type' not in group.attrs:
            raise ValueError("Was this hdf5 really created from our scripts?"
                             "Internal organization not recognized. Groups "
                             "should have a 'type' attribute.")

        if group.attrs['type'] == 'volume':
            extract_volume(group, args.outname)
        elif group.attrs['type'] == 'streamlines':
            if args.save_connectivity:
                extract_connectivity(group, args.outname)
            else:
                extract_sft(group, args.outname)
        else:
            raise ValueError("Was this hdf5 really created from our scripts?"
                             "Group type not recognized. Expected 'volume' or "
                             "'streamlines'.")


if __name__ == '__main__':
    main()
