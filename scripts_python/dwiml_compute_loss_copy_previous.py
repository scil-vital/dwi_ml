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

import torch.nn.functional
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, assert_inputs_exist
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions, normalize_directions
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.utils.direction_getters import add_direction_getter_args

CHOICES = ['cosine-regression', 'l2-regression', 'cosine-plus-l2-regression',
           'sphere-classification']


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input_sft',
                   help="Input tractogram to use as target.")
    add_direction_getter_args(p, dropout_arg=False, gaussian_fisher_args=False)

    p.add_argument('--use_0_for_first_dir', action='store_true',
                   help="If set, previous direction for the first point will "
                        "be [0,0,0]. Else, first point's loss is skipped.")
    sub = p.add_mutually_exclusive_group()
    sub.add_argument(
        '--step_size', type=float, metavar='s',
        help="Resample all streamlines to this step size (in mm). "
             "Default = None.")
    sub.add_argument(
        '--compress', action='store_true',
        help="If set, compress streamlines.")
    add_reference_arg(p)

    return p


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # 1. Prepare direction getter
    if args.dg_key not in CHOICES:
        raise NotImplementedError("Script ot ready for this choice of "
                                  "direction getter")
    dg_cls = keys_to_direction_getters[args.dg_key]
    input_size = 1  # Fake, we won't use the forward method.
    if 'regression' in args.dg_key:
        dg = dg_cls(input_size, dropout=0,
                    normalize_targets=args.normalize_targets,
                    normalize_outputs=args.normalize_outputs)
    else:
        dg = dg_cls(input_size, dropout=0)

    # 2. Load SFT
    assert_inputs_exist(p, args.input_sft, args.reference)
    sft = load_tractogram_with_reference(p, args, args.input_sft)
    sft.to_vox()

    if args.step_size:
        sft = resample_streamlines_step_size(sft, args.step_size)
    else:
        sft = compress_sft(sft)

    streamlines = [torch.as_tensor(s) for s in sft.streamlines]
    directions = compute_directions(streamlines)

    if args.use_0_for_first_dir:
        zeros = torch.zeros(3)
        directions = [torch.vstack((zeros, d)) for d in directions]

    # 3. Prepare targets and fake outputs.
    targets = [d[1:, :] for d in directions]
    outputs = [d[:-1, :] for d in directions]
    outputs = torch.vstack(outputs)
    targets = torch.vstack(targets)
    if args.dg_key == 'sphere-classification':
        # Prepare logits_per_class
        # Just the index of the class as a one-hot
        classes = dg.torch_sphere.find_closest(outputs)
        outputs = torch.nn.functional.one_hot(classes.to(dtype=torch.long))
    else:  # regression
        if args.normalize_outputs:
            outputs = normalize_directions(outputs) * args.normalize_outputs

    # 4. Compute loss
    with torch.no_grad():
        loss, n = dg.compute_loss(outputs, targets)

    print("{} loss function, averaged over all {} points in the chosen SFT, "
          "is: {}.".format(args.dg_key, n, loss))


if __name__ == '__main__':
    main()
