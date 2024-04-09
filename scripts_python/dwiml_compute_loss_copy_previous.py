#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
One difficulty of choosing a good loss function for tractography is that
streamlines have the particularity of being smooth.

Printing the average loss function for a given dataset when we simply copy the
previous direction.

    Target :=  SFT.streamlines' directions[1:]
    Y := Previous directions.
    loss = DirectionGetter(Target, Y)
"""
import argparse
import logging

import torch.nn.functional

from dwi_ml.io_utils import add_resample_or_compress_arg
from dwi_ml.models.projects.copy_previous_dirs import CopyPrevDirModel
from dwi_ml.models.utils.direction_getters import add_direction_getter_args, \
    check_args_direction_getter
from dwi_ml.testing.testers import Tester
from dwi_ml.testing.utils import add_args_testing_subj_hdf5
from dwi_ml.testing.visu_loss import run_all_visu_loss
from dwi_ml.testing.visu_loss_utils import prepare_args_visu_loss, visu_checks

CHOICES = ['cosine-regression', 'l2-regression', 'sphere-classification',
           'smooth-sphere-classification', 'cosine-plus-l2-regression']


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    add_args_testing_subj_hdf5(p, ask_input_group=False,
                               ask_streamlines_group=False)
    prepare_args_visu_loss(p)
    p.add_argument('--skip_first_point', action='store_true',
                   help="If set, do not compute the loss at the first point "
                        "of the streamline. \nElse (default) compute it with "
                        "previous dir = 0.")
    add_resample_or_compress_arg(p)
    add_direction_getter_args(p)

    return p


def main():
    p = prepare_arg_parser()
    args = p.parse_args()
    logging.getLogger().setLevel(level=args.logging)

    # Checks
    if args.out_dir is None:
        p.error("Please specify out_dir, as there is not experiment path for "
                "this fake experiment.")
    names = visu_checks(args, p)

    # Device
    device = (torch.device('cuda') if torch.cuda.is_available() and
              args.use_gpu else None)

    # 1. Prepare fake model
    dg_args = check_args_direction_getter(args)
    model = CopyPrevDirModel(args.dg_key, dg_args, args.skip_first_point,
                             args.step_size, args.compress)
    model.set_context('visu')

    # 2. Load data through the tester
    tester = Tester(model, args.subj_id, args.hdf5_file, args.subset,
                    args.batch_size, device)

    run_all_visu_loss(tester, model, args, names)


if __name__ == '__main__':
    main()
