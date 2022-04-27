#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs

NB_PREVIOUS_DIRS = 4


def test_previous_dirs(script_runner):
    print("\n"
          "Unit test: previous dirs\n"
          "------------------------")

    # Short streamline: length 1
    streamline1 = torch.tensor([[3.2, 31.2, 45.7]], dtype=torch.float32)

    # Streamline of length 2
    streamline2 = torch.tensor([[5, 5, 5],
                                [6, 7, 8]], dtype=torch.float32)

    # Long streamline: length 12
    sub_streamline = torch.tensor([[1, 1, 1],
                                   [2, 2, 2],
                                   [6, 6, 6],
                                   [4, 4, 4]], dtype=torch.float32)
    streamline3 = torch.cat((sub_streamline, sub_streamline, sub_streamline))

    # Testing previous dirs with various lengths of streamlines
    for streamline in [streamline1, streamline2, streamline3]:
        print("Streamline is:\n{}".format(streamline))

        # The function is meant to receive a batch of streamlines. Creating a
        # batch of 1 streamline.
        streamlines = [streamline]

        # 1. Computing directions (same formula as in code)
        streamline_dirs = [torch.as_tensor(s[1:] - s[:-1],
                                           dtype=torch.float32)
                           for s in streamlines]
        print("Streamlines dirs are:\n{}".format(streamline_dirs))

        tmp = compute_n_previous_dirs(streamline_dirs, NB_PREVIOUS_DIRS)

        # Returns the batch of previous dirs. Taking result for the first (and
        # only streamline)
        prev_dirs = tmp[0]
        print("The {} previous dirs at each point are:\n{}"
              .format(NB_PREVIOUS_DIRS, prev_dirs))

        # There should be one set of previous dirs per point
        assert len(prev_dirs) == len(streamline)

        # There should be 3*NB_PREVIOUS_DIRS coordinates per point.
        assert prev_dirs.shape[1] == 3 * NB_PREVIOUS_DIRS
