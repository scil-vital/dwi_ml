# -*- coding: utf-8 -*-
from typing import List, Union

import torch
from torch import Tensor
from torch.nn.utils.rnn import (PackedSequence, pack_sequence)

from dwi_ml.models.main_models import ModelAbstract


class EmbeddingAbstract(ModelAbstract):
    def __init__(self, input_size: int, output_size: int = 128):
        """
        Params
        -------
        input_size: int
            Size of input data or, in the case of a sequence, of each data
            point. If working with streamlines, probably a multiple of 3
            ([x,y,z] for each direction).
        output_size: int
            Size of output data or of each output data point.
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size

    @property
    def attributes(self):
        # We need real int types, not numpy.int64, not recognized by json
        # dumps.
        attributes = {
            'input_size': int(self.input_size),
            'output_size': int(self.output_size),
        }
        return attributes

    @property
    def hyperparameters(self):
        return {}

    def forward(self, inputs):
        raise NotImplementedError


class NNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int):
        super().__init__(input_size, output_size)
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.relu = torch.nn.ReLU()

    @property
    def attributes(self):
        attrs = super().attributes  # type: dict
        attrs.update({
            'key': 'nn_embedding'
        })
        return attrs

    def forward(self, inputs: Tensor):
        self.log.debug("Embedding: running Neural networks' forward")

        # Calling forward.
        result = self.linear(inputs)
        result = self.relu(result)

        return result


class NoEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int = None):
        if output_size is None:
            output_size = input_size
        if input_size != output_size:
            self.log.debug("Identity embedding should have input_size == "
                           "output_size. Not stopping now but this won't work "
                           "if your data does not follow the shape you are "
                           "suggesting.")

        super().__init__(input_size, output_size)
        self.identity = torch.nn.Identity()

    def forward(self, inputs: Tensor = None):
        self.log.debug("Embedding: running identity's forward")
        # toDo. Should check that input size = self.input_size but we don't
        #  know how the data is organized. Probably inputs.shape[0]?
        result = self.identity(inputs)
        return result

    @property
    def attributes(self):
        attrs = super().attributes  # type: dict
        attrs.update({
            'key': 'no_embedding'
        })
        return attrs


class CNNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.cnn_layer = torch.nn.Conv3d

    @property
    def attributes(self):
        params = super().attributes  # type: dict
        other_parameters = {
            'layers': 'non-defined-yet',
            'key': 'cnn_embedding'
        }
        return params.update(other_parameters)

    def forward(self, inputs: Tensor):
        raise NotImplementedError


keys_to_embeddings = {'no_embedding': NoEmbedding,
                      'nn_embedding': NNEmbedding,
                      'cnn_embedding': CNNEmbedding}
