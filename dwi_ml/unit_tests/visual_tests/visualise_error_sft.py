# -*- coding: utf-8 -*-
import argparse
import itertools
import json
import logging
import os

import numpy as np
import torch
from dipy.io.streamline import save_tractogram
from scilpy.io.streamlines import load_tractogram_with_reference
from scilpy.io.utils import add_reference_arg, add_overwrite_arg, add_bbox_arg

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.models.main_models import MainModelAbstract
from dwi_ml.arg_utils import get_resample_or_compress_arg, add_logging_arg
from dwi_ml.testing.utils import prepare_dataset_one_subj

blue = [2., 75., 252.]


def build_argparser_visu_error(skip_exp=False):
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__)
    if not skip_exp:
        p.add_argument(
            'experiment_path',
            help='Path to the directory containing the experiment.\n'
                 '(Should contain a model subdir with a file '
                 'parameters.json\n and a file best_model_state.pkl.)')
    p.add_argument(
        'hdf5_file',
        help="Path to the hdf5 file.")
    p.add_argument(
        'subj_id',
        help="Subject id whose' input data to use. However, for easier "
             "usage \n(easier picking of one bundle / one streamline of "
             "interest), the streamlines group \nwill not be picked from the "
             "hdf5 file.")
    p.add_argument(
        'input_sft',
        help="A small tractogram; a bundle. We will save a tractogram with \n"
             "one additional streamline beside each, showing the learned \n"
             "direction error. Between each point, the new streamline will \n"
             "come back to the true point. Colors will be : \n"
             "- blue for the error.\n"
             "- a gradient from green to pink for the true streamline: \n"
             "  green is the seed (we do not split streamlines here).")
    p.add_argument(
        'out_sft',
        help="Output tractogram filename.")
    p.add_argument(
        '--pick_one', action='store_true',
        help="If set, pick only one streamline from the input SFT")
    p.add_argument('--subset', default='testing',
                   choices=['training', 'validation', 'testing'],
                   help="Subject id should probably come come the "
                        "'testing' set but you can\n modify this to "
                        "'training' or 'validation'.")
    get_resample_or_compress_arg(p)
    add_reference_arg(p)
    add_bbox_arg(p)
    add_overwrite_arg(p)
    add_logging_arg(p)

    return p


def prepare_batch_visu_error(p, args, model: MainModelAbstract,
                             check_data_loader=True):
    """
    Loads the batch input from one subject.
    Loads the batch streamlines from an SFT rather than from the hdf5.
    """
    # Load SFT, possibly pick one streamline
    sft = load_tractogram_with_reference(p, args, args.input_sft)

    if len(sft) > 1 and args.pick_one:
        streamline_ids = np.random.randint(0, len(sft), size=1)
        logging.info("Tractogram contained more than one streamline. "
                     "Picking any one: #{}.".format(streamline_ids))

        sft = sft[streamline_ids]

    # Mimic hdf5 creation: resample / compress to user's definition.
    sft = resample_or_compress(sft, args.step_size, args.compress)

    # Mimic DataLoader:
    #   1) On streamlines: resample / compress (again).
    #      (We do not revert, split, or add noise.)
    # Load batch loader's info
    if check_data_loader:
        params_filename = os.path.join(args.experiment_path,
                                       "parameters_latest.json")
        with open(params_filename, 'r') as json_file:
            params = json.load(json_file)
        input_group = params["Batch loader params"]["input_group_name"]
        step_size = params["Batch loader params"]["step_size"]
        compress = params["Batch loader params"]["compress"]

        if not (step_size == args.step_size and compress == args.compress):
            sft = resample_or_compress(sft, step_size, compress)
    else:
        input_group = args.input_group

    sft.to_vox()
    sft.to_corner()
    streamlines = [torch.as_tensor(s) for s in sft.streamlines]
    if not model.direction_getter.add_eos:
        # We don't use the last coord because it does not have an
        # associated target direction.
        streamlines = [s[:-1, :] for s in streamlines]

    #   2) On inputs (done in dataloader / trainer)
    logging.info("Loading subject.")
    subset, subj_idx = prepare_dataset_one_subj(
        args.hdf5_file, args.subj_id, lazy=False, cache_size=False,
        subset_name=args.subset)
    group_idx = subset.volume_groups.index(input_group)
    streamlines_minus_one = [s[:-1, :] for s in streamlines]
    batch_input, _ = model.prepare_batch_one_input(
        streamlines_minus_one, subset, subj_idx, group_idx)
    logging.info("Loaded and prepared associated batch input for all {} "
                 "streamlines.".format(len(batch_input)))

    return sft, streamlines, batch_input


def save_output_with_ref(out_dirs, args, sft):
    """
    Normalizes directions.
    Saves the model-learned streamlines together with the input streamlines:
        - This streamline is created by starting at the real position at each
        point, then advancing in the learned direction.
        - Between two points, we always add a point going back to the real
        position to view difference better. This means that the learned
        streamlines is twice the length of the real streamline.
    """
    epsilon = torch.tensor(0.000005)
    if args.step_size is not None:
        _step_size_vox = args.step_size / sft.space_attributes[2]
    else:
        _step_size_vox = 0

    out_streamlines = []
    color_x = []
    color_y = []
    color_z = []
    for i, s in enumerate(sft.streamlines):
        this_s_len = len(s)

        # Normalizing directions to step_size
        out_dir = out_dirs[i]

        if _step_size_vox is not None:
            out_dir /= torch.max(epsilon,
                                 torch.linalg.norm(out_dir, dim=1)[:, None])
            out_dir *= _step_size_vox
        out_dir = out_dir.numpy()

        assert len(out_dir) == this_s_len - 1

        # output : Starts on first point.
        #          Then each point = true previous point + learned dir
        #          in between each point, comes back to correct point.
        tmp = [[s[p] + out_dir[p, :], s[p+1]]
               for p in range(this_s_len - 1)]
        out_streamline = \
            [s[0]] + list(itertools.chain.from_iterable(tmp))
        out_streamline = out_streamline[:-1]
        this_s_len2 = len(out_streamline)

        # Two points per point except first and last
        assert this_s_len2 == this_s_len * 2 - 2

        out_streamlines.extend([s, out_streamline])

        # Data per point: Add a color to differentiate both streamlines.
        # Ref streamline = blue
        all_x_blue = [[blue[0]]] * this_s_len
        all_y_blue = [[blue[1]]] * this_s_len
        all_z_blue = [[blue[2]]] * this_s_len
        # Learned streamline = from green to pink
        ranging_2 = [[i / this_s_len2 * 252.] for i in range(this_s_len2)]

        color_x.extend([all_x_blue, ranging_2],)
        color_y.extend([all_y_blue, [[150.]] * this_s_len2])
        color_z.extend([all_z_blue, ranging_2])

    assert len(out_streamlines) == len(sft) * 2
    data_per_point = {
        'color_x': color_x,
        'color_y': color_y,
        'color_z': color_z
    }
    sft = sft.from_sft(out_streamlines, sft, data_per_point)

    print("Saving out tractogram to {}: \n"
          "  - In blue: The original streamline.\n"
          "  - From dark green (start) to pink: displacement of "
          "estimation at each time point.".format(args.out_sft))
    save_tractogram(sft, args.out_sft, bbox_valid_check=False)
