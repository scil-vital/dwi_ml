# -*- coding: utf-8 -*-
import logging

import torch
from torch import Tensor


class EmbeddingAbstract(torch.nn.Module):
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
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        # We need real int types, not numpy.int64, not recognized by json
        # dumps.
        params = {
            'input_size': int(self.input_size),
            'output_size': int(self.output_size),
        }
        return params

    def forward(self, inputs):
        raise NotImplementedError


class NNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int):
        super().__init__(input_size, output_size)
        self.linear = torch.nn.Linear(self.input_size, self.output_size)
        self.relu = torch.nn.ReLU()

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'nn_embedding'
        })
        return params

    def forward(self, inputs: Tensor):
        # Calling forward.
        result = self.linear(inputs)
        result = self.relu(result)
        return result


class NoEmbedding(EmbeddingAbstract):
    def __init__(self, input_size, output_size: int = None):
        if output_size is None:
            output_size = input_size
        if input_size != output_size:
            raise ValueError("Identity embedding should have input_size == "
                             "output_size but you gave {} and {}"
                             .format(input_size, output_size))

        super().__init__(input_size, output_size)
        self.identity = torch.nn.Identity()

    def forward(self, inputs: Tensor = None):
        #  Should check that input size = self.input_size but we don't
        #  know how the data is organized. Letting user be responsible.
        result = self.identity(inputs)
        return result

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'no_embedding'
        })
        return params


class CNNEmbedding(EmbeddingAbstract):
    def __init__(self, input_size: int, output_size: int):
        super().__init__(input_size, output_size)
        self.cnn_layer = torch.nn.Conv3d

    @property
    def params(self):
        params = super().params  # type: dict
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
