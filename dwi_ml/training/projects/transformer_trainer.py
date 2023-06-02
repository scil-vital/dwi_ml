# -*- coding: utf-8 -*-
import logging
from typing import List

import torch

from dwi_ml.models.projects.transforming_tractography import AbstractTransformerModel
from dwi_ml.training.batch_samplers import DWIMLBatchIDSampler
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.projects.trainers_for_generation import \
    DWIMLTrainerForTrackingOneInput


class TransformerTrainer(DWIMLTrainerForTrackingOneInput):
    def __init__(self, **kwargs):
        """
        See Super for parameter description. No additional parameters here.
        """
        super().__init__(**kwargs)

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
