# -*- coding: utf-8 -*-
import logging
import math

from typing import List

import torch
from torch import nn
from torch import Tensor
from torch.nn import functional as F

from dwi_ml.models.main_models import MainModelAbstract


class PositionalEncoding(nn.Module):
    """ Modified from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.htm://pytorch.org/tutorials/beginner/transformer_tutorial.html  # noqa E504
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]
        x = self.dropout(x)
        x = x.permute(1, 0, 2)
        return x


class ModelAE(MainModelAbstract):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, an RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """
    def __init__(self, kernel_size, latent_space_dims,
                 experiment_name: str,
                 # Target preprocessing params for the batch loader + tracker
                 step_size: float = None,
                 compress_lines: float = False,
                 # Other
                 log_level=logging.root.level):
        super().__init__(experiment_name, step_size, compress_lines, log_level)

        # Embedding size, could be defined by the user ?
        self.embedding_size = 32
        # Embedding layer
        self.embedding = nn.Sequential(
            *(nn.Linear(3, self.embedding_size),
              nn.ReLU()))

        # Positional encoding layer
        self.pos_encoding = PositionalEncoding(
            self.embedding_size, max_len=(256))
        # Transformer encoder layer
        layer = nn.TransformerEncoderLayer(
            self.embedding_size, 4, batch_first=True)

        # Transformer encoder
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.decoder = nn.TransformerEncoder(layer, 4)

        self.reconstruction_loss = torch.nn.MSELoss()

        self.pad = torch.nn.ReflectionPad1d(1)
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims

        self.fc1 = torch.nn.Linear(8192,
                                   self.latent_space_dims)  # 8192 = 1024*8
        self.fc2 = torch.nn.Linear(self.latent_space_dims, 8192)

        self.fc3 = torch.nn.Linear(self.embedding_size, 3)

        def pre_pad(m):
            return torch.nn.Sequential(self.pad, m)

        """
        Encode convolutions
        """
        self.encod_conv1 = pre_pad(
            torch.nn.Conv1d(self.embedding_size, 32,
                            self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv2 = pre_pad(
            torch.nn.Conv1d(32, 64, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv3 = pre_pad(
            torch.nn.Conv1d(64, 128, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv4 = pre_pad(
            torch.nn.Conv1d(128, 256, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv5 = pre_pad(
            torch.nn.Conv1d(256, 512, self.kernel_size, stride=2, padding=0)
        )
        self.encod_conv6 = pre_pad(
            torch.nn.Conv1d(512, 1024, self.kernel_size, stride=1, padding=0)
        )

        """
        Decode convolutions
        """
        self.decod_conv1 = pre_pad(
            torch.nn.Conv1d(1024, 512, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl1 = torch.nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv2 = pre_pad(
            torch.nn.Conv1d(512, 256, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl2 = torch.nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv3 = pre_pad(
            torch.nn.Conv1d(256, 128, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl3 = torch.nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv4 = pre_pad(
            torch.nn.Conv1d(128, 64, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl4 = torch.nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv5 = pre_pad(
            torch.nn.Conv1d(64, 32, self.kernel_size, stride=1, padding=0)
        )
        self.upsampl5 = torch.nn.Upsample(
            scale_factor=2, mode="linear", align_corners=False
        )
        self.decod_conv6 = pre_pad(
            torch.nn.Conv1d(32, 32,
                            self.kernel_size, stride=1, padding=0)
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

    @classmethod
    def _load_params(cls, model_dir):
        p = super()._load_params(model_dir)
        p['kernel_size'] = 3
        p['latent_space_dims'] = 32
        return p

    def forward(self,
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

    def encode(self, x):
        # x: list of tensors
        x = torch.stack(x)

        x = self.embedding(x) * math.sqrt(self.embedding_size)
        x = self.pos_encoding(x)
        x = self.encoder(x)

        x = torch.swapaxes(x, 1, 2)

        h1 = F.relu(self.encod_conv1(x))
        h2 = F.relu(self.encod_conv2(h1))
        h3 = F.relu(self.encod_conv3(h2))
        h4 = F.relu(self.encod_conv4(h3))
        h5 = F.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = (h6.shape[1], h6.shape[2])

        # Flatten
        h7 = h6.view(-1, self.encoder_out_size[0] * self.encoder_out_size[1])

        fc1 = self.fc1(h7)

        return fc1

    def decode(self, z):

        fc = self.fc2(z)
        fc_reshape = fc.view(
            -1, self.encoder_out_size[0], self.encoder_out_size[1]
        )
        h1 = F.relu(self.decod_conv1(fc_reshape))
        h2 = self.upsampl1(h1)
        h3 = F.relu(self.decod_conv2(h2))
        h4 = self.upsampl2(h3)
        h5 = F.relu(self.decod_conv3(h4))
        h6 = self.upsampl3(h5)
        h7 = F.relu(self.decod_conv4(h6))
        h8 = self.upsampl4(h7)
        h9 = F.relu(self.decod_conv5(h8))
        h10 = self.upsampl5(h9)
        h11 = self.decod_conv6(h10)

        h11 = h11.permute(0, 2, 1)

        h12 = self.decoder(h11)

        x = self.fc3(h12)

        return x.permute(0, 2, 1)

    def compute_loss(self, model_outputs, targets, average_results=True):
        targets = torch.stack(targets)
        targets = torch.swapaxes(targets, 1, 2)
        mse = self.reconstruction_loss(model_outputs, targets)

        return mse, 1
