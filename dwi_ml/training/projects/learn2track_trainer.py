# -*- coding: utf-8 -*-
import logging

import torch

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput, DWIMLTrainerForTrackingOneInput

logger = logging.getLogger('trainer_logger')


class Learn2TrackTrainer(DWIMLTrainerForTrackingOneInput):
    """
    Trainer for Learn2Track. Nearly the same as in dwi_ml, but we add the
    clip_grad parameter to avoid exploding gradients, typical in RNN.
    """

    def __init__(self, clip_grad: float = None, **kwargs):
        """
        Init trainer.

        Additional values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        """
        super().__init__(**kwargs)

        self.clip_grad = clip_grad

    @property
    def params_for_checkpoint(self):
        params = super().params_for_checkpoint
        params.update({
            'clip_grad': self.clip_grad
        })
        return params

    def _prepare_checkpoint_info(self) -> dict:
        checkpoint_state = super()._prepare_checkpoint_info()
        checkpoint_state['params_for_init'].update({
            'clip_grad': self.clip_grad
        })

        return checkpoint_state

    def fix_parameters(self):
        """
        In our case, clipping gradients to avoid exploding gradients in RNN
        """
        if self.clip_grad is not None:
            total_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_grad)
            if torch.isnan(total_norm):
                raise ValueError("Exploding gradients. Experiment failed.")

    def prepare_model_to_track(self, lines, ids_per_subj):
        # Running the beginning of the sequence to get the hidden state.
        batch_inputs = self.batch_loader.load_batch_inputs(lines, ids_per_subj)

        logger.debug('*** Computing forward propagation for N={} steps'
                     .format(self.tracking_phase_nb_steps_init))
        if self.model.forward_uses_streamlines:
            # Possibly computing directions twice (during forward and loss)
            # but ok, shouldn't be too heavy. Easier to deal with multiple
            # projects' requirements by sending whole streamlines rather
            # than only directions.
            _, hidden_state = self.model(batch_inputs, lines)
        else:
            _, hidden_state = self.model(batch_inputs)

        return hidden_state
