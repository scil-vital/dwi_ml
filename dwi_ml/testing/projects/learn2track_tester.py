# -*- coding: utf-8 -*-
import itertools

import torch
from torch.nn.utils.rnn import unpack_sequence

from dwi_ml.testing.testers import TesterOneInput


class TesterPackedSequence(TesterOneInput):
    def combine_batches(self, losses, outputs):
        if len(losses) == 1:
            return losses[0], unpack_sequence(outputs[0])

        losses = torch.cat(losses)
        outputs = [unpack_sequence(batch) for batch in outputs]
        outputs = list(itertools.chain.from_iterable(outputs))
        return losses, outputs
