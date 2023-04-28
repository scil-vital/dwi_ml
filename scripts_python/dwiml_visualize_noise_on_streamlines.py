#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One difficulty of choosing a good loss function for tractography is that
streamlines have the particularity of being smooth.

Printing the average loss function for a given dataset when we simply copy the
previous direction.

    Target :=  SFT.streamlines's directions[1:]
    Y := Previous directions.
    loss = DirectionGetter(Target, Y)
"""
import argparse
import logging

import torch.nn.functional
from dipy.io.streamline import save_tractogram

from scilpy.io.utils import assert_inputs_exist, assert_outputs_exist, \
    add_overwrite_arg

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.data.processing.utils import add_noise_to_tensor
from dwi_ml.testing.utils import add_args_testing_subj_hdf5, \
    prepare_dataset_one_subj
from dwi_ml.training.utils.batch_loaders import add_args_batch_loader


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_args_testing_subj_hdf5(p)
    p.add_argument('streamlines_group',
                   help="Streamline group to use as SFT for the given "
                        "subject in the hdf5.")
    add_args_batch_loader(p)
    p.add_argument('out_name',
                   help='Name of the modified SFT.')
    add_overwrite_arg(p)

    return p


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # Verify output names
    assert_inputs_exist(p, args.hdf5_file)
    assert_outputs_exist(p, args, args.out_name)

    # Create dataset
    logging.info("Loading subject {} from hdf5.".format(args.subj_id))
    subset, subj_idx = prepare_dataset_one_subj(
        args.hdf5_file, args.subj_id, subset_name=args.subset,
        volume_groups=[], streamline_groups=[args.streamlines_group],
        lazy=False, cache_size=None)

    # Load SFT as in dataloader, except we don't loop on many subject,
    # we don't verify streamline ids (loading all), and we don't split /
    # reverse streamlines. But we resample / compress + add noise.
    logging.info("Loading its streamlines as SFT.")
    streamline_group_idx = subset.streamline_groups.index(
        args.streamlines_group)
    subj_data = subset.subjs_data_list.get_subj_with_handle(subj_idx)
    subj_sft_data = subj_data.sft_data_list[streamline_group_idx]
    sft = subj_sft_data.as_sft()

    sft = resample_or_compress(sft, args.step_size, args.compress)
    sft.to_vox()
    sft.to_corner()

    logging.debug("            Adding noise {}"
                  .format(args.noise_gaussian_size_forward))
    streamlines = [torch.as_tensor(s) for s in sft.streamlines]
    streamlines = add_noise_to_tensor(streamlines,
                                      args.noise_gaussian_size_forward,
                                      device=None)
    streamlines = [s.tolist() for s in streamlines]
    sft = sft.from_sft(streamlines, sft,
                       sft.data_per_point, sft.data_per_streamline)

    # 3. Save colored SFT
    save_tractogram(sft, args.out_name)


if __name__ == '__main__':
    main()
