# -*- coding: utf-8 -*-
import logging
from typing import List

import torch

from dwi_ml.models.projects.transforming_tractography import AbstractTransformerModel
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput


class TransformerTrainer(DWIMLTrainerOneInput):
    def __init__(self,
                 model: AbstractTransformerModel, experiments_path: str,
                 experiment_name: str,
                 batch_sampler: DWIMLBatchIDSampler,
                 batch_loader: DWIMLBatchLoaderOneInput,
                 learning_rate: float = 0.001, weight_decay: float = 0.01,
                 use_radam: bool = False, betas: List[float] = None,
                 max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: int = 1000,
                 patience: int = None, nb_cpu_processes: int = 0,
                 use_gpu: bool = False, comet_workspace: str = None,
                 comet_project: str = None, from_checkpoint: bool = False,
                 log_level=logging.root.level):
        """
        See Super for parameter description. No additional parameters here.
        """
        super().__init__(model, experiments_path, experiment_name,
                         batch_sampler, batch_loader,
                         learning_rate, weight_decay,
                         use_radam, betas, max_epochs,
                         max_batches_per_epoch_training,
                         max_batches_per_epoch_validation,
                         patience, nb_cpu_processes, use_gpu,
                         comet_workspace, comet_project,
                         from_checkpoint, log_level)

    @classmethod
    def init_from_checkpoint(
            cls, model: AbstractTransformerModel, experiments_path,
            experiment_name, batch_sampler: DWIMLBatchIDSampler,
            batch_loader: DWIMLBatchLoaderOneInput,
            checkpoint_state: dict, new_patience, new_max_epochs,
            log_level):
        """
        During save_checkpoint(), checkpoint_state.pkl is saved. Loading it
        back offers a dict that can be used to instantiate an experiment and
        set it at the same state as previously. (Current_epoch is updated +1).
        """

        # Use super's method but return this transformer trainer as 'cls'.
        experiment = super(cls, cls).init_from_checkpoint(
            model, experiments_path, experiment_name,
            batch_sampler, batch_loader,
            checkpoint_state, new_patience, new_max_epochs, log_level)

        return experiment

    def run_model(self, batch_inputs, batch_streamlines):
        dirs = self.model.format_directions(batch_streamlines)

        # Formatting the previous dirs for all points.
        n_prev_dirs = self.model.format_previous_dirs(dirs, self.device)

        # Not keeping the last point: only useful to get the last direction
        # (last target), but won't be used as an input.
        if n_prev_dirs is not None:
            n_prev_dirs = [s[:-1] for s in n_prev_dirs]

        try:
            # Apply model. This calls our model's forward function
            # (the hidden states are not used here, neither as input nor
            # outputs. We need them only during tracking).
            model_outputs, _ = self.model(batch_inputs, n_prev_dirs,
                                          self.device)
        except RuntimeError:
            # Training RNNs with variable-length sequences on the GPU can
            # cause memory fragmentation in the pytorch-managed cache,
            # possibly leading to "random" OOM RuntimeError during
            # training. Emptying the GPU cache seems to fix the problem for
            # now. We don't do it every update because it can be time
            # consuming.
            torch.cuda.empty_cache()
            model_outputs, _ = self.model(batch_inputs, n_prev_dirs,
                                          self.device)

        # Returning the directions too, to be re-used in compute_loss
        # later instead of computing them twice.
        return model_outputs, dirs
