# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
"""
import logging
from typing import Union, List

import torch
from torch.nn.utils.rnn import unpack_sequence

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput

logger = logging.getLogger('trainer_logger')


class Learn2TrackTrainer(DWIMLTrainerOneInput):
    """
    Trainer for Learn2Track. Nearly the same as in dwi_ml, but we add the
    clip_grad parameter to avoid exploding gradients, typical in RNN.
    """

    def __init__(self,
                 model: Learn2TrackModel, experiments_path: str,
                 experiment_name: str,
                 batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLBatchLoaderOneInput,
                 learning_rates: List = None, weight_decay: float = 0.01,
                 optimizer='Adam', max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None, patience_delta: float = 1e-6,
                 nb_cpu_processes: int = 0, use_gpu: bool = False,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, clip_grad: float = None,
                 log_level=logging.WARNING, use_radam: bool = None,
                 learning_rate: float = None):
        """
        Init trainer.

        Additional values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        """
        super().__init__(
            model=model, experiments_path=experiments_path,
            experiment_name=experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader, learning_rates=learning_rates,
            weight_decay=weight_decay, optimizer=optimizer,
            use_radam=use_radam, max_epochs=max_epochs,
            max_batches_per_epoch_training=max_batches_per_epoch_training,
            max_batches_per_epoch_validation=max_batches_per_epoch_validation,
            patience=patience, patience_delta=patience_delta,
            nb_cpu_processes=nb_cpu_processes, use_gpu=use_gpu,
            comet_workspace=comet_workspace, comet_project=comet_project,
            from_checkpoint=from_checkpoint,
            log_level=log_level, learning_rate=learning_rate)

        self.clip_grad = clip_grad
        self.real_hidden_state_memory = None  # For n-step training
        self.tmp_hidden_state_memory = None

    @property
    def params_for_checkpoint(self):
        params = super().params_for_checkpoint
        params.update({
            'clip_grad': self.clip_grad
        })
        return params

    @classmethod
    def init_from_checkpoint(
            cls, model: Learn2TrackModel, experiments_path, experiment_name,
            batch_sampler: DWIMLBatchIDSampler,
            batch_loader: DWIMLBatchLoaderOneInput,
            checkpoint_state: dict, new_patience,
            new_max_epochs, log_level):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this learn2track trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            model, experiments_path, experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, new_patience, new_max_epochs, log_level)

        return experiment

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
