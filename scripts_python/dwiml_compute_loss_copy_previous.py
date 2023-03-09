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

import numpy as np
import torch.nn.functional
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, assert_inputs_exist

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions, normalize_directions
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    convert_dirs_to_class
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.utils import add_resample_or_compress_arg, resample_or_compress

CHOICES = ['cosine-regression', 'l2-regression', 'sphere-classification',
           'smooth-sphere-classification', 'cosine-plus-l2-regression']


def prepare_arg_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('input_sft',
                   help="Input tractogram to use as target.")
    p.add_argument(
        '--dg_key', metavar='key', default='cosine-regression',
        choices=CHOICES,
        help="Model for the direction getter layer. Regression or "
             "classification only.")

    add_resample_or_compress_arg(p)
    add_reference_arg(p)

    return p


def main():
    p = prepare_arg_parser()
    args = p.parse_args()

    # 1. Prepare direction getter
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
    sft = resample_or_compress(sft, args.step_size, args.compress)

    # To tensor
    sft.to_vox()
    sft.to_corner()
    streamlines = [torch.as_tensor(s, dtype=torch.float32)
                   for s in sft.streamlines]

    # 3. Prepare targets and outputs: both out of directions.
    # Similar to direction_getter.prepare_targets:
    # A) Compute directions. Shift + add a fake first direction.
    targets = compute_directions(streamlines)

    # B) Output at each point = copy previous dir.
    #    - No prev dir at first point = we use random.
    #    - Last dir is not used.
    rand = torch.as_tensor(np.random.rand(3), dtype=torch.float32)
    outputs = [torch.vstack((rand, t[:-1, :])) for t in targets]

    # 4. Format targets and outputs.
    if args.dg_key == 'sphere-classification':
        targets = convert_dirs_to_class(targets, dg.torch_sphere)
        outputs = convert_dirs_to_class(outputs, dg.torch_sphere)
        targets = torch.hstack(targets)
        outputs = torch.hstack(outputs)

        # Prepare logits_per_class
        outputs = torch.nn.functional.one_hot(outputs.to(dtype=torch.long))
    else:  # regression
        targets = torch.vstack(targets)
        if args.normalize_outputs:
            outputs = normalize_directions(outputs) * args.normalize_outputs
        outputs = torch.vstack(outputs)

    # 4. Compute loss
    with torch.no_grad():
        loss, n = dg.compute_loss(outputs, targets)

    print("{} loss function, averaged over all {} points in the chosen SFT, "
          "is: {}.".format(args.dg_key, n, loss))


if __name__ == '__main__':
    main()
