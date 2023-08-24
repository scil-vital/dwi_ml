# -*- coding: utf-8 -*-
from typing import Tuple, Union, List

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
    def __init__(self, nb_features_in: int,
                 activation: str = None, key: str = ''):
        """
        Params
        -------
        input_size: int
            Size of each data point.
            Ex: MRI features * nb neighbors for flattened MRI data.
            Ex: 3D coordinates [x,y,z] for streamlines.
        output_size: int
            Size of each output data point.
        activation: str
            Name of the activation layer. Currently, only accepted values are
            None or 'ReLu'.
        """
        super().__init__()
        self.nb_features_in = nb_features_in
        self.key = key
        self.activation = activation
        self.activation_layer = None
        if self.activation is not None:
            if self.activation.lower() == 'relu':
                self.activation_layer = torch.nn.ReLU()
            else:
                raise NotImplementedError("Activation function not "
                                          "recognized for embedding layer.")

    def forward(self, inputs):
        raise NotImplementedError


class NNEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in, nb_features_out: int):
        super().__init__(nb_features_in, activation='ReLu',
                         key='nn_embedding')
        self.nb_features_out = nb_features_out
        self.linear = torch.nn.Linear(self.nb_features_in, self.nb_features_out)

    def forward(self, atensor: Tensor):
        # Calling forward.
        atensor = self.linear(atensor)
        atensor = self.activation_layer(atensor)
        return atensor


class NoEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in, nb_features_out=None):
        super().__init__(nb_features_in, activation=None, key='no_embedding')

        if nb_features_out is not None:
            assert nb_features_in == nb_features_out
            
        self.identity = torch.nn.Identity()

    def forward(self, inputs: Tensor = None):
        #  Should check that input size = self.input_size but we don't
        #  know how the data is organized. Letting user be responsible.
        result = self.identity(inputs)
        return result


class CNNEmbedding(EmbeddingAbstract):
    def __init__(self, nb_features_in: int, nb_filters: List[int],
                 kernel_sizes: List[int], image_shape: Tuple[int, int, int]):
        """
        Applies a 3D convolution. For now: a single layer.

        Parameters
        ----------
        nb_features_in: int
            Size should refer to the number of features per voxel. (Contrary to
            other embeddings, where data in each neighborhood is flattened and
            input_size is thus nb_features * nb_neighboors).
        nb_filters: List[int]
            Size of the output = number of out_channels = number of filters.
            One value per layer.
        kernel_sizes: List[int]
            Size of the kernel for each layer (will be a 3D [k, k, k] kernel).
        image_shape: (int, int, int)
            Size of the image.
        """
        super().__init__(nb_features_in, activation='ReLu', key='cnn_embedding')

        self.in_image_shape = np.asarray(image_shape)

        padding = 0
        stride = 1
        dilation = 1
        self.cnn_layers = []
        in_shape = self.in_image_shape
        for i in range(len(nb_filters)):
            cnn_layer = torch.nn.Conv3d(
                nb_features_in, nb_filters[i],
                kernel_size=kernel_sizes[i], padding=padding,
                stride=stride, dilation=dilation)

            # Adding module explicitly as is it not name.
            self.add_module("cnn_{}".format(i), cnn_layer)
            self.cnn_layers.append(cnn_layer)

            # Computing values for next layer
            # Output size formula is given here:
            # https://pytorch.org/docs/stable/generated/torch.nn.Conv3d.html
            numerator = \
                in_shape + 2 * padding - dilation * (kernel_sizes[i] - 1) - 1

            out_shape = np.floor(numerator / stride + 1).astype(int)

            # For next layer:
            in_shape = out_shape
            nb_features_in = nb_filters[i]

        self.out_image_shape = out_shape
        self.out_flattened_size = int(np.prod(self.out_image_shape) * nb_filters[-1])

    def forward(self, x: Tensor):
        """
        Expected inputs shape: (batch size, x, y, z, channels)
          (We will reorder to torch's input shape.)
        Outputs shape: (batch size, x2, y2, x2, c2)
        """
        # Torch:
        #   Note that depth is first, but we can pretend it is (N, C, X, Y, Z).
        #   3D operation is the same.
        # Input size = (N, C1, D1, H1, W1)  --> (B, C1, X1, Y1, Z1)
        # Output size = (N, C2, D2, H2, W2) --> (B, C2, X2, Y2, Z2)
        #     N = Batch size.
        #     C = Number of channels
        #     D = Depth of image
        #     H = Height of image
        #     W = Width of image
        assert x.shape[-1] == self.nb_features_in
        assert np.array_equal(x.shape[1:4], self.in_image_shape), \
            "Expecting inputs of shape {} ({} channels per voxel), but "\
            "received {}.".format(self.in_image_shape, self.nb_features_in,
                                  x.shape[2:])

        x = torch.permute(x, (0, 4, 1, 2, 3))
        # Current shape = (B, C, X, Y, Z)
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
            x = self.activation_layer(x)

        x = torch.permute(x, (0, 2, 3, 4, 1))
        # Current shape = (B, X2, Y2, Z2, C2)
        x = torch.flatten(x, start_dim=1, end_dim=4)
        # Final shape: (B, X2*Y2*Z2*C2)

        return x


keys_to_embeddings = {'no_embedding': NoEmbedding,
                      'nn_embedding': NNEmbedding,
                      'cnn_embedding': CNNEmbedding}
