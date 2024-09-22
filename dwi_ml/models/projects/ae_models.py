# -*- coding: utf-8 -*-
import logging
from typing import List

import torch

from dwi_ml.models.main_models import MainModelAbstract


class ResBlock1d(torch.nn.Module):

    def __init__(self, channels, stride=1):
        super(ResBlock1d, self).__init__()

        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(channels, channels, kernel_size=1,
                            stride=stride, padding=0),
            torch.nn.BatchNorm1d(channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(channels, channels, kernel_size=3,
                            stride=stride, padding=1),
            torch.nn.BatchNorm1d(channels),
            torch.nn.GELU(),
            torch.nn.Conv1d(channels, channels, 1,
                            1, 0),
            torch.nn.BatchNorm1d(channels))

    def forward(self, x):
        identity = x
        xp = self.block(x)

        return xp + identity


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

        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
        self.reconstruction_loss = torch.nn.MSELoss(reduction="sum")

        self.fc1 = torch.nn.Linear(8192,
                                   self.latent_space_dims)  # 8192 = 1024*8
        self.fc2 = torch.nn.Linear(self.latent_space_dims, 8192)

        """
        Encode convolutions
        """
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv1d(3, 64, self.kernel_size, stride=2, padding=1),
            torch.nn.GELU(),
            ResBlock1d(64),
            ResBlock1d(64),
            ResBlock1d(64),
            torch.nn.Conv1d(64, 64, self.kernel_size, stride=2, padding=1),
            torch.nn.GELU(),
            ResBlock1d(64),
            ResBlock1d(64),
            ResBlock1d(64),
            torch.nn.Conv1d(64, 256, self.kernel_size, stride=2, padding=1),
            torch.nn.GELU(),
            ResBlock1d(256),
            ResBlock1d(256),
            ResBlock1d(256),
            torch.nn.Conv1d(256, 256, self.kernel_size, stride=2, padding=1),
            torch.nn.GELU(),
            ResBlock1d(256),
            ResBlock1d(256),
            ResBlock1d(256),
            torch.nn.Conv1d(256, 1024, self.kernel_size, stride=2, padding=1),
            torch.nn.GELU(),
            ResBlock1d(1024),
            ResBlock1d(1024),
            ResBlock1d(1024),
            torch.nn.Conv1d(1024, 1024, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(1024),
            ResBlock1d(1024),
            ResBlock1d(1024),
        )

        """
        Decode convolutions
        """
        self.decoder = torch.nn.Sequential(
            ResBlock1d(1024),
            ResBlock1d(1024),
            ResBlock1d(1024),
            torch.nn.GELU(),
            torch.nn.Conv1d(
                1024, 1024, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(1024),
            ResBlock1d(1024),
            ResBlock1d(1024),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                1024, 256, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(256),
            ResBlock1d(256),
            ResBlock1d(256),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                256, 256, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(256),
            ResBlock1d(256),
            ResBlock1d(256),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                256, 64, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(64),
            ResBlock1d(64),
            ResBlock1d(64),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                64, 64, self.kernel_size, stride=1, padding=1),
            torch.nn.GELU(),
            ResBlock1d(64),
            ResBlock1d(64),
            ResBlock1d(64),
            torch.nn.Upsample(scale_factor=2, mode="linear",
                              align_corners=False),
            torch.nn.Conv1d(
                64, 3, self.kernel_size, stride=1, padding=1),
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
        x = torch.swapaxes(x, 1, 2)

        x = self.encoder(x)
        self.encoder_out_size = (x.shape[1], x.shape[2])

        # Flatten
        h7 = x.view(-1, self.encoder_out_size[0] * self.encoder_out_size[1])

        fc1 = self.fc1(h7)
        return fc1

    def decode(self, z):
        fc = self.fc2(z)
        fc_reshape = fc.view(
            -1, self.encoder_out_size[0], self.encoder_out_size[1]
        )
        z = self.decoder(fc_reshape)
        return z

    def compute_loss(self, model_outputs, targets, average_results=True):
        targets = torch.stack(targets)
        targets = torch.swapaxes(targets, 1, 2)
        mse = self.reconstruction_loss(model_outputs, targets)

        # loss_function_vae
        # See Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kld = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        # kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # kld = torch.sum(kld_element).__mul__(-0.5)

        return mse, 1
