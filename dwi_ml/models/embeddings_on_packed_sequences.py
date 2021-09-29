# -*- coding: utf-8 -*-
from typing import List, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import (PackedSequence, pack_sequence)

from dwi_ml.data.packed_sequences import (unpack_sequence,
                                          unpack_tensor_from_indices)
from dwi_ml.models.embeddings_on_tensors import (NNEmbedding as NNe,
                                                 NoEmbedding as Noe,
                                                 CNNEmbedding as CNNe)


class NNEmbedding(NNe):
    def __init__(self, input_size, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, inputs: PackedSequence):
        self.log.debug("Embedding: running Neural networks' forward")

        # Unpacking
        inputs_tensor = inputs.data

        result = super().forward(inputs_tensor)

        # Packing
        # Not using pack_sequence because we want to keep the same info as
        # the inputs (nb of feature has changed, but not the number of inputs
        # -- i.e. streamlines)
        result = PackedSequence(result, inputs.batch_sizes,
                                inputs.sorted_indices,
                                inputs.unsorted_indices)
        return result


class NoEmbedding(Noe):
    def __init__(self, input_size, output_size: int = None):
        super().__init__(input_size, output_size)

    def forward(self, inputs: PackedSequence = None):
        # As this forward does nothing, it takes any input type as input,
        # no need to change anything for a packedSequence.
        result = super().forward(inputs)
        return result


class CNNEmbedding(CNNe):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)

    def forward(self, inputs: PackedSequence):
        raise NotImplementedError


keys_to_embeddings = {'no_embedding': NoEmbedding,
                      'nn_embedding': NNEmbedding,
                      'cnn_embedding': CNNEmbedding}
