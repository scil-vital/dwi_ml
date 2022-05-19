#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.data.packed_sequences import (unpack_sequence,
                                          unpack_tensor_from_indices)
logging.basicConfig(level='INFO')


def test_packing_unpacking():

    directions1 = torch.tensor(np.array([[1, 0, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]], dtype='float32'))
    directions2 = torch.tensor(np.array([[2, 0, 0],
                                         [2, 2, 2],
                                         [0, 2, 0]], dtype='float32'))
    streamlines = [directions1, directions2]

    logging.info('Testing packing and unpacking')
    logging.info("   Input before packing: {}".format(streamlines))

    # Packing batch
    packed_sequence = pack_sequence(streamlines, enforce_sorted=False)
    inputs_tensor = packed_sequence.data
    logging.info("   Packed: {}".format(inputs_tensor))

    # Unpacking batch, technique 1
    result = unpack_sequence(packed_sequence)
    logging.info("   Unpacked technique 1: {}".format(result))
    for s in range(len(streamlines)):
        assert torch.equal(result[s], streamlines[s])

    # Unpacking batch, technique 2
    indices = unpack_sequence(packed_sequence, get_indices_only=True)
    result = unpack_tensor_from_indices(inputs_tensor, indices)
    logging.info("   Unpacked technique 2: {}".format(result))
    for s in range(len(streamlines)):
        assert torch.equal(result[s], streamlines[s])


if __name__ == '__main__':
    test_packing_unpacking()
