#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

logging.getLogger().setLevel(level='INFO')


def test_packing_unpacking():

    streamline1 = torch.tensor(np.array([[1, 0, 0],
                                         [1, 1, 1],
                                         [0, 1, 0]], dtype='float32'))
    streamline2 = torch.tensor(np.array([[2, 0, 0],
                                         [2, 2, 2],
                                         [0, 2, 0]], dtype='float32'))
    streamlines = [streamline1, streamline2]

    logging.info('Testing packing and unpacking')
    logging.info("   Input before packing: {}".format(streamlines))

    # Packing batch
    packed_sequence = pack_sequence(streamlines, enforce_sorted=False)
    inputs_tensor = packed_sequence.data
    logging.info("   Packed: {}".format(inputs_tensor))

    # Unpacking batch
    result, _ = pad_packed_sequence(packed_sequence, batch_first=True)
    logging.info("   Unpacked technique: {}".format(result))
    for s in range(len(streamlines)):
        assert torch.equal(result[s], streamlines[s])


if __name__ == '__main__':
    test_packing_unpacking()
