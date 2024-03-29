#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from dwi_ml.data.processing.streamlines.post_processing import compute_triu_connectivity_from_blocs


def test_connectivity():
    # Ex: Volume is 16 x 16

    # Streamline starting at the lowest left side to the highest right side.
    streamline = [[0, 0], [15.9, 15.9]]
    streamlines = [streamline, streamline]

    # to a 4x4 matrix, should have two values from "ROI" 0 to "ROI" 15.
    expected_m = np.zeros((16, 16), dtype=int)
    expected_m[0, 15] = 2
    #  expected_m[15, 0] = 2  ----> but triu
    print("Expected connectivity matrix: {}".format(expected_m))

    m, _, _ = compute_triu_connectivity_from_blocs(streamlines, (16, 16), (4, 4))
    print("Got {}".format(m))
    assert np.array_equal(m, expected_m)


if __name__ == '__main__':
    test_connectivity()
