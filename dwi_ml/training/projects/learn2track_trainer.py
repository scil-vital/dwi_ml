# -*- coding: utf-8 -*-
"""
author: Philippe Poulin (philippe.poulin2@usherbrooke.ca),
        refactored by Emmanuelle Renauld
"""
import logging
from typing import Union, List

import numpy as np
import torch

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
                 learning_rate: float = 0.001,
                 weight_decay: float = 0.01,
                 use_radam: bool = False, betas: List[float] = None,
                 max_epochs: int = 10,
                 max_batches_per_epoch_training: int = 1000,
                 max_batches_per_epoch_validation: Union[int, None] = 1000,
                 patience: int = None,
                 nb_cpu_processes: int = 0, use_gpu: bool = False,
                 comet_workspace: str = None, comet_project: str = None,
                 from_checkpoint: bool = False, clip_grad: float = None,
                 log_level=logging.WARNING):
        """
        Init trainer.

        Additionnal values compared to super:
        clip_grad : float
            The value to which to clip gradients after the backward pass.
            There is no good value here. Default: 1000.
        """
        super().__init__(
            model=model, experiments_path=experiments_path,
            experiment_name=experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader, learning_rate=learning_rate,
            weight_decay=weight_decay, use_radam=use_radam, betas=betas,
            max_epochs=max_epochs,
            max_batches_per_epoch_training=max_batches_per_epoch_training,
            max_batches_per_epoch_validation=max_batches_per_epoch_validation,
            patience=patience, nb_cpu_processes=nb_cpu_processes,
            use_gpu=use_gpu, comet_workspace=comet_workspace,
            comet_project=comet_project, from_checkpoint=from_checkpoint,
            log_level=log_level)

        self.clip_grad = clip_grad
        self.weight_visualizor = {
            "previous_dirs": [],
            "features_current_point": [],
            "features_mean_neighborhood": [],
            "neighborhood_mean_features": [],
            "hidden_state": []
        }

    def train_one_epoch(self, epoch):
        """
        Adding an option to visualise weights.
        """
        super().train_one_epoch(epoch)

        if self.model.input_embedding_key == 'no_embedding':
            self._save_weights_for_visualization()
        else:
            self.weight_visualizor = "Input embedding was used. I wouldn't " \
                                     "know how to interpret this. Could try " \
                                     "instead to analyse the embedding layer."

    def _save_weights_for_visualization(self):
        # For the signification of weights:
        # https://medium.com/@andre.holzner/lstm-cells-in-pytorch-fab924a78b1c
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        # We check the first layer weights, hoping they show us how the input
        # is used. Other techniques exist, google "Feature importance".
        # Using abs: direction is not important.

        # 1. Weights coming from the inputs to the first layer are
        # (W_ii | W_if | W_ig | W_io), of shape(4 * hidden_size, input_size)
        weights = self.model.rnn_model.rnn_layers[0].weight_ih_l0
        weights = np.abs(weights.cpu().detach().numpy())
        nb_features = self.model.nb_features
        nb_neighbors = self.model.nb_neighbors + 1
        # weights.shape[1] is the total nb features (* neighbors, + PD).

        # 1.A) Previous dirs.
        if self.model.nb_previous_dirs > 0:
            nb_features_pd = self.model.prev_dirs_embedding_size
            mean_pd_weights = np.mean(weights[:, -nb_features_pd:])
            self.weight_visualizor['previous_dirs'].append(mean_pd_weights)
            weights = weights[:, :-nb_features_pd]

        # 1. B) Feature importance
        # If there is a neighborhood, using only current coordinate
        # (first point).
        self.weight_visualizor["features_current_point"].append(
            np.mean(weights[:, :nb_features], axis=0))

        # 1. C) Feature importance
        # (averaged throughout neighborhood points).
        if nb_neighbors > 1:
            # Coords were a list of neighborhood points.
            # So interpolation gives: features at each point.
            # Data is organized as [[f1 f2 f3]_n1 [f1 f2 f3]_n2 ]
            assert nb_features == int((weights.shape[1]) / nb_neighbors), \
                "Nb features in model param: {} * nb neighbors {} != nb " \
                "weights: {}".format(nb_features,  nb_neighbors,
                                     weights.shape[1])

            self.weight_visualizor["features_mean_neighborhood"].append(
                [np.mean(weights[:, f::nb_features])
                 for f in range(nb_features)])

        # 1. D) Coordinate importance (averaged throughout features)
        if nb_neighbors > 1:
            bins = list(range(0, weights.shape[1], nb_features))
            self.weight_visualizor["neighborhood_mean_features"].append(
                np.mean(np.add.reduceat(weights, bins, axis=1) / nb_features,
                        axis=0))

        # 2. Weights coming from the hidden state. Can we use them?
        # (W_hi|W_hf|W_hg|W_ho), of shape (4*hidden_size, hidden_size).
        weights = self.model.rnn_model.rnn_layers[0].weight_hh_l0
        weights = np.abs(weights.cpu().detach().numpy())
        self.weight_visualizor['hidden_state'].append(np.mean(weights))

    @property
    def params_for_checkpoint(self):
        # We do not need the model_uses_streamlines params, we know it is true
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

        experiment.weight_visualizor = \
            checkpoint_state['current_states']['weights_vizualizor']

        return experiment

    def _prepare_checkpoint_info(self) -> dict:
        checkpoint_state = super()._prepare_checkpoint_info()
        checkpoint_state['params_for_init'].update({
            'clip_grad': self.clip_grad
        })

        checkpoint_state['current_states'].update({
            'weights_vizualizor': self.weight_visualizor
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
