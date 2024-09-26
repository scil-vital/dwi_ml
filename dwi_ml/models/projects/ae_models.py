# -*- coding: utf-8 -*-
import logging
from typing import List

import torch
from torch.nn import functional as F

from dwi_ml.models.main_models import MainModelAbstract


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

        self.pad = torch.nn.ReflectionPad1d(1)

        self.post_encoding_hooks = []

        def pre_pad(m):
            return torch.nn.Sequential(self.pad, m)

        self.fc1 = torch.nn.Linear(8192,
                                   self.latent_space_dims)  # 8192 = 1024*8
        self.fc2 = torch.nn.Linear(self.latent_space_dims, 8192)

        """
        Encode convolutions
        """
        self.encod_conv1 = pre_pad(
            torch.nn.Conv1d(3, 32, self.kernel_size, stride=2, padding=0)
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
            torch.nn.Conv1d(32, 3, self.kernel_size, stride=1, padding=0)
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
        encoded = self.encode(input_streamlines)

        for hook in self.post_encoding_hooks:
            hook(encoded)

        model_outputs = self.decode(encoded)
        return model_outputs

    def encode(self, x):
        # X input shape is (batch_size, nb_points, 3)
        if isinstance(x, list):
            x = torch.stack(x)

        x = torch.swapaxes(x, 1, 2) # input of the network should be (N, 3, nb_points)

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

        return h11

    def compute_loss(self, model_outputs, targets, average_results=True):
        print("COMPARISON\n")
        targets = torch.stack(targets)
        targets = torch.swapaxes(targets, 1, 2)
        print(targets[0, :, 0:5])
        print(model_outputs[0, :, 0:5])
        reconstruction_loss = torch.nn.MSELoss(reduction="sum")
        mse = reconstruction_loss(model_outputs, targets)

        # loss_function_vae
        # See Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # kld = -0.5 * torch.sum(1 + self.logvar - self.mu.pow(2) - self.logvar.exp())
        # kld_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
        # kld = torch.sum(kld_element).__mul__(-0.5)

        return mse, 1

    def register_hook_post_encoding(self, hook):
        self.post_encoding_hooks.append(hook)