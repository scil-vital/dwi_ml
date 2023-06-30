#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

from dwi_ml.data.processing.streamlines.post_processing import compute_triu_connectivity


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

    m = compute_triu_connectivity(streamlines, (16, 16), (4, 4))
    print("Got {}".format(m))
    assert np.array_equal(m, expected_m)

    m = compute_triu_connectivity(streamlines, (16, 16), (4, 4),
                                  to_sparse_tensor=True)
    m2 = m.to_dense().numpy().astype(int)
    print("Converting to sparse and back to dense: {}".format(m2))
    assert np.array_equal(m2, expected_m)
