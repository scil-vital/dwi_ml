# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
import torch
from torch import Tensor

"""
Hint: To use on packed sequences:

    # Unpacking
    inputs_tensor = inputs.data

    result = embedding(inputs_tensor)

    # Packing back
    # Not using pack_sequence because we want to keep the same info as
    # the inputs (nb of feature has changed, but not the number of inputs
    # -- i.e. streamlines)
    result = PackedSequence(result, inputs.batch_sizes,
                            inputs.sorted_indices,
                            inputs.unsorted_indices)
"""


class EmbeddingAbstract(torch.nn.Module):
    def __init__(self, nb_features_in: int, nb_features_out: int,
                 key: str = ''):
        """
        Params
        -------
        input_size: int
            Size of each data point.
            Ex: MRI features * nb neighbors for flattened MRI data.
            Ex: 3D coordinates [x,y,z] for streamlines.
        output_size: int
            Size of each output data point.
        """
        super().__init__()
        self.nb_features_in = nb_features_in
        self.nb_features_out = nb_features_out
        self.key = key

    @property
    def params_for_checkpoint(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        # We need real int types, not numpy.int64, not recognized by json
        # dumps.
        params = {
            'nb_features_in': int(self.nb_features_in),
            'nb_features_out': int(self.nb_features_out),
            'key': self.key
        }
        return params

    def forward(self, inputs):
        raise NotImplementedError


class NNEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in, nb_features_out: int):
        super().__init__(nb_features_in, nb_features_out, key='nn_embedding')
        self.linear = torch.nn.Linear(self.nb_features_in, self.nb_features_out)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs: Tensor):
        # Calling forward.
        result = self.linear(inputs)
        result = self.relu(result)
        return result


class NoEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in, nb_features_out: int = None):
        if nb_features_out is None:
            nb_features_out = nb_features_in
        if nb_features_in != nb_features_out:
            raise ValueError("Identity embedding should have input_size == "
                             "output_size but you gave {} and {}"
                             .format(nb_features_in, nb_features_out))

        super().__init__(nb_features_in, nb_features_out, key='no_embedding')
        self.identity = torch.nn.Identity()

    def forward(self, inputs: Tensor = None):
        #  Should check that input size = self.input_size but we don't
        #  know how the data is organized. Letting user be responsible.
        result = self.identity(inputs)
        return result


class CNNEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in: int, nb_features_out: int,
                 kernel_size: int, image_shape: Tuple):
        """
        Applies a 3D convolution. For now: a single layer.

        Parameters
        ----------
        nb_features_in: int
            Size should refer to the number of features per voxel. (Contrary to
            other embeddings, where data in each neighborhood is flattened and
            input_size is thus nb_features * nb_neighboors).
        nb_features_out: int
            Size of the output = number of out_channels = number of filters.
        kernel_size: int
            Size of the kernel (will be a 3D [k, k, k] kernel).
        image_shape: (int, int, int)
            Size of the image.
        """
        super().__init__(nb_features_in, nb_features_out, key='cnn_embedding')

        # Torch:
        # Input size = (N, C1, D1, H1, W1)
        # Output size = (N, C2, D2, H2, W2)
        #     N = Batch size.
        #     C = Number of channels
        #     D = Depth of image
        #     H = Height of image
        #     W = Width of image
        # kernel size = one int, or a tuple of 3 ints.
        self.in_image_shape = list(image_shape)
        self.cnn_layer = torch.nn.Conv3d(
            nb_features_in, nb_features_out, kernel_size=kernel_size)

        out_image_shape = [0] * 3
        padding = 0
        stride = 1
        dilation=1
        # Using default stride=1, padding=0, dilation=1, etc.
        # Output size formula is given here:
        # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
        for i in range(3):
            numerator = self.in_image_shape[i] + 2 * padding - \
                        dilation * (kernel_size - 1) - 1
            out_image_shape[i] = np.floor(numerator / stride + 1)
        self.out_flattened_size = np.prod(out_image_shape)

    def forward(self, inputs: Tensor):
        assert inputs.shape[1] == self.nb_features_in
        assert np.all(inputs.shape[2:] == self.in_image_shape)
        raise NotImplementedError


keys_to_embeddings = {'no_embedding': NoEmbedding,
                      'nn_embedding': NNEmbedding,
                      'cnn_embedding': CNNEmbedding}
