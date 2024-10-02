# -*- coding: utf-8 -*-
import logging
from typing import List

import torch

from dwi_ml.models.main_models import MainModelAbstract


class Permute(torch.nn.Module):
    """This module returns a view of the tensor input with its dimensions permuted.
    From https://github.com/pytorch/vision/blob/main/torchvision/ops/misc.py#L308  # noqa E501

    Args:
        dims (List[int]): The desired ordering of dimensions
    """

    def __init__(self, dims: List[int]):
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.permute(x, self.dims)


class LayerNorm1d(torch.nn.LayerNorm):
    """ Layer normalization module. Uses the same normalization as torch.nn.LayerNorm
    but with the input tensor permuted before and after the normalization. This is
    necessary because torch.nn.LayerNorm normalizes the last dimension of the input
    tensor, but we want to normalize the middle dimension to fit with convolutional layers.

    From https://github.com/pytorch/vision/blob/main/torchvision/models/convnext.py # noqa E501
    """
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = torch.nn.functional.layer_norm(
            x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 1)
        return x


class ResBlock1d(torch.nn.Module):
    """ ConvNext residual block.

    Uses a 1D convolution with kernel size 7 and groups=channels to ensure that
    each channel is processed independently. The block is composed of a layer
    normalization, a GELU activation function, a linear layer with 4 times the
    number of channels, a GELU activation function, and a linear layer with the
    same number of channels as the input. The output is added to the input
    tensor.
    """

    def __init__(self, channels, stride=1, norm=LayerNorm1d):
        """ Constructor

        Parameters
        ----------
        channels : int
            Number of channels in the input tensor.
        stride : int
            Stride of the convolution.
        norm : torch.nn.Module
            Normalization layer to use.
        """

        super(ResBlock1d, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=7, groups=channels,
                            stride=stride, padding=3, padding_mode='reflect'),
            norm(channels),
            Permute((0, 2, 1)),
            torch.nn.Linear(
                in_features=channels, out_features=4 * channels, bias=True),
            torch.nn.GELU(),
            torch.nn.Linear(
                in_features=channels * 4, out_features=channels, bias=True),
            Permute((0, 2, 1)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass.
        """

        identity = x
        x = self.block(x)

        return x + identity


class ModelConvNextAE(MainModelAbstract):
    """ ConvNext autoencoder model, modified to work with 1D data.

    The model is composed of an encoder and a decoder. The encoder is composed
    of a series of 1D convolutions and residual blocks. The decoder is composed
    of residual blocks, upsampling layers, and 1D convolutions. The decoder is
    much smaller than the encoder to reduce the number of parameters and
    enforce a well defined latent space.

    References:
    [1]: Liu, Z., Mao, H., Wu, C. Y., Feichtenhofer, C., Darrell, T., & Xie, S.
    (2022). A convnet for the 2020s. In Proceedings of the IEEE/CVF conference
    on computer vision and pattern recognition (pp. 11976-11986).
    """

    def __init__(
        self,
        kernel_size,
        latent_space_dims,
        experiment_name: str,
        nb_points: int = None,
        step_size: float = None,
        compress_lines: float = False,
        # Other
        log_level=logging.root.level
    ):
        super().__init__(
            experiment_name,
            step_size=step_size,
            nb_points=nb_points,
            compress_lines=compress_lines,
            log_level=log_level
        )

        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
        self.reconstruction_loss = torch.nn.MSELoss(reduction="sum")

        """
        Encode convolutions. Follows the same structure as the original
        ConvNext-Tiny architecture, but with 1D convolutions.
        """
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 32, self.kernel_size+1, stride=1, padding=1,
                            padding_mode='reflect'),
            ResBlock1d(32),
            ResBlock1d(32),
            ResBlock1d(32),
            torch.nn.Conv1d(32, 64, self.kernel_size, stride=2, padding=0,
                            padding_mode='reflect'),
            ResBlock1d(64),
            ResBlock1d(64),
            ResBlock1d(64),
            torch.nn.Conv1d(64, 128, self.kernel_size, stride=2, padding=0,
                            padding_mode='reflect'),
            ResBlock1d(128),
            ResBlock1d(128),
            ResBlock1d(128),
            torch.nn.Conv1d(128, 256, self.kernel_size, stride=2, padding=0,
                            padding_mode='reflect'),
            ResBlock1d(256),
            ResBlock1d(256),
            ResBlock1d(256),
            torch.nn.Conv1d(256, 512, self.kernel_size, stride=2, padding=0,
                            padding_mode='reflect'),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            ResBlock1d(512),
            torch.nn.Conv1d(512, 1024, self.kernel_size, stride=2, padding=0,
                            padding_mode='reflect'),
            ResBlock1d(1024),
            ResBlock1d(1024),
            ResBlock1d(1024),
        )
        """
        Latent space
        """

        self.fc1 = torch.nn.Linear(8192,
                                   self.latent_space_dims)  # 8192 = 1024*8
        self.fc2 = torch.nn.Linear(self.latent_space_dims, 8192)

        """
        Decode convolutions. Uses upsampling and 1D convolutions instead of
        transposed convolutions to avoid checkerboard artifacts.
        """
        self.decoder = torch.nn.Sequential(
            ResBlock1d(1024),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                1024, 512, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
            ResBlock1d(512),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                512, 256, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
            ResBlock1d(256),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                256, 128, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
            ResBlock1d(128),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                128, 64, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
            ResBlock1d(64),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                64, 32, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
            ResBlock1d(32),
            torch.nn.Conv1d(
                32, 3, self.kernel_size+1, stride=1, padding=1,
                padding_mode='reflect'),
        )

    @property
    def params_for_checkpoint(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        # p = super().params_for_checkpoint()
        p = {'kernel_size': self.kernel_size,
             'latent_space_dims': self.latent_space_dims,
             'experiment_name': self.experiment_name}
        return p

    def forward(
        self,
        input_streamlines: List[torch.tensor],
    ):
        """Run the model on a batch of sequences.

        Parameters
        ----------
        input_streamlines: List[torch.tensor],
            Batch of streamlines. Only used if previous directions are added to
            the model. Used to compute directions; its last point will not be
            used.

        Returns
        -------
        model_outputs : List[Tensor]
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
        """

        x = self.decode(self.encode(input_streamlines))
        return x

    def encode(self, x: List[torch.Tensor]) -> torch.Tensor:
        """ Encode the input data.

        Parameters
        ----------
        x : list of tensors
            List of input tensors.

        Returns
        -------
        z : torch.Tensor
            Input data encoded to the latent space.
        """

        # x: list of tensors
        if isinstance(x, list):
            x = torch.stack(x)
        x = torch.swapaxes(x, 1, 2)

        x = self.encoder(x)
        self.encoder_out_size = (x.shape[1], x.shape[2])

        # Flatten
        h7 = x.reshape(-1, self.encoder_out_size[0] * self.encoder_out_size[1])

        z = self.fc1(h7)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """ Decode the input data.

        Parameters
        ----------
        z : torch.Tensor
            Input data in the latent space.

        Returns
        -------
        x_hat : torch.Tensor
            Decoded data.
        """

        fc = self.fc2(z)
        fc_reshape = fc.view(
            -1, self.encoder_out_size[0], self.encoder_out_size[1]
        )
        x_hat = self.decoder(fc_reshape)
        return x_hat

    def compute_loss(self, model_outputs, targets, average_results=True):
        """Compute the loss of the model.

        Parameters
        ----------
        model_outputs : List[Tensor]
            Model outputs.
        targets : List[Tensor]
            Target data.
        average_results : bool
            If True, the loss will be averaged across the batch.

        Returns
        -------
        loss : Tensor
            Loss value.
        """

        targets = torch.stack(targets)
        targets = torch.swapaxes(targets, 1, 2)
        mse = self.reconstruction_loss(model_outputs, targets)

        return mse, 1
