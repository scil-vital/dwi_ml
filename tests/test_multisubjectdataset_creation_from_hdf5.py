#!/usr/bin/env python
import argparse
import logging
import os
from os import path
from argparse import RawTextHelpFormatter

import numpy as np
from dipy.io.stateful_tractogram import Origin, Space, StatefulTractogram
from dipy.io.streamline import save_tractogram
from nibabel import Nifti1Image
from torch.utils.data.dataloader import DataLoader

from dwi_ml.data.dataset.dataset import (MultiSubjectDataset, 
                                         LazyMultiSubjectDataset)
from dwi_ml.data.dataset.batch_sampler import (BatchSampler)


def parse_args():
    """
    Convert "raw" streamlines from a hdf5 to torch data or to PackedSequences
    via our MultiSubjectDataset class. Test some properties.

    The MultiSubjectDataset is used during training, in the trainer_abstract.
    """
    parser = argparse.ArgumentParser(description=str(parse_args.__doc__),
                                     formatter_class=RawTextHelpFormatter)
    p.add_argument('hdf5_filename',
                   help='Path to the .hdf5 dataset. Should contain both your '
                        'training subjects and validation subjects.')
    p.add_argument('training_subjs_filename',
                   help='Txt file containing the list of subjects to use for '
                        'training. One subject per line. All subjects should '
                        'exist as training subjects in the hdf5 dataset.')
    p.add_argument('validation_subjs_filename',
                   help='Txt file containing the list of subjects used for '
                        'validation. One subject per line. Can be None. All '
                        'subjects should exist as validation subjects in the '
                        'hdf5 dataset.')

    return parser.parse_args()


def save_batches():
    """
    Saving batches on disk to open them and see what was sampled
    """
    for i, (streamlines, tid_to_slice) in enumerate(batches):
        for tid, array_slice in tid_to_slice.items():
            # Don't use the last coord because it is used only to compute last target direction
            flattened_slice_coords = np.concatenate([s[:-1] for s in streamlines[array_slice]], axis=0)

            data_volume = dataset._get_tractodata_volume(tid)  # torch.Tensor

            input_dv = dataset._get_tractodata(tid).input_dv
            fname = 'subject_{}_batch_{}.tck'.format(input_dv.subject_id, i)
            output = os.path.join(args.output_folder, fname)
            logging.info("Saving batch of streamlines to {}".format(output))
            ref = Nifti1Image(data_volume.numpy(), input_dv.affine_vox2rasmm)
            sft = StatefulTractogram(streamlines[array_slice], ref,
                                     Space.VOX,
                                     origin=Origin.NIFTI)
            save_tractogram(sft, output, bbox_valid_check=False)


if __name__ == '__main__':
    args = parse_args()

    if not path.exists(args.training_subjs_filename):
        raise ValueError("The training subjects list ({}) was not found!"
                         .format(args.training_subjs_filename))
    if args.validation_subjs_filename and \
            not path.exists(args.validation_subjs_filename):
        raise ValueError("The validation subjects list ({}) was not found!"
                         .format(args.validation_subjs_filename))
    if not path.exists(args.hdf5_filename):
        raise ValueError("The validation subjects list ({}) was not found!"
                         .format(args.hdf5_filename))

    # Options tested:
    rng = np.random.RandomState(seed=1234)

    # Creating dataset
    fake_dataset = MultiSubjectDataset(
        args.training_subjs_filename, rng)

    fake_lazy_dataset = LazyMultiSubjectDataset(
        args.training_subjs_filename, rng)

    # Using datasets
    #fake_dataset.load()

    # Sampling batch
    #sampler = BatchSampler(data_source=fake_dataset)

    # Loading batch
    #dataloader = DataLoader(fake_dataset,
    #                        batch_sampler=sampler,
    #                        num_workers=0,
    #                        collate_fn=fake_dataset.collate_fn)
    #batches = [next(iter(dataloader)) for i in range(5)]

    #save_batches()

    # Delete references to avoid a weird Exception when force closing the hdf5
    # file `ImportError: sys.meta_path is None, Python is likely shutting down`
    #del dataloader
    #del sampler
    del fake_dataset
