# -*- coding: utf-8 -*-
import torch
from torch import Tensor
from typing import List

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


def unpad_sequence(padded_sequences: Tensor, lengths: Tensor,
                   batch_first: bool = False, ) -> List[Tensor]:
    """
    Method copied from torch 1.11, waiting for Beluga to upgrade available
    wheels to 1.11 (current max is 1.10, so we can't install torch 1.11).
    """
    unpadded_sequences = []

    if not batch_first:
        padded_sequences.transpose_(0, 1)

    max_length = padded_sequences.shape[1]
    idx = torch.arange(max_length)

    for seq, length in zip(padded_sequences, lengths):
        mask = idx < length
        unpacked_seq = seq[mask]
        unpadded_sequences.append(unpacked_seq)

    return unpadded_sequences


def unpack_sequence(packed_sequences: PackedSequence) -> List[Tensor]:
    """
    Method copied from torch 1.11, waiting for Beluga to upgrade available
    wheels to 1.11 (current max is 1.10, so we can't install torch 1.11).
    """

    padded_sequences, lengths = pad_packed_sequence(packed_sequences,
                                                    batch_first=True)
    unpacked_sequences = unpad_sequence(padded_sequences, lengths,
                                        batch_first=True)
    return unpacked_sequences
