# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.utils import prepare_tracking_mask
from dwi_ml.training.with_generation.trainer import \
    DWIMLTrainerForTrackingOneInput

logger = logging.getLogger('trainer_logger')


class Learn2TrackTrainer(DWIMLTrainerForTrackingOneInput):
    """
    Trainer for Learn2Track. Nearly the same as in dwi_ml, but we add the
    clip_grad parameter to avoid exploding gradients, typical in RNN.
    """
    model: Learn2TrackModel

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

    def propagate_multiple_lines(self, lines: List[torch.Tensor], ids_per_subj):
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        # Running the beginning of the streamlines to get the hidden states
        # (using one less point. The next will be done during propagation).
        tmp_lines = [line[:-1, :] for line in lines]
        inputs = self.batch_loader.load_batch_inputs(tmp_lines, ids_per_subj)
        _, hidden_states = self.model(inputs, tmp_lines, return_hidden=True)
        del tmp_lines

        def update_memory_after_removing_lines(can_continue: np.ndarray, _):
            nonlocal hidden_states
            hidden_states = self.model.remove_lines_in_hidden_state(
                hidden_states, can_continue)

        def get_dirs_at_last_pos(_lines: List[torch.Tensor], n_last_pos):
            nonlocal hidden_states

            n_last_pos = [pos[None, :] for pos in n_last_pos]
            batch_inputs = self.batch_loader.load_batch_inputs(n_last_pos,
                                                               ids_per_subj)

            model_outputs, hidden_states = self.model(
                batch_inputs, _lines, hidden_recurrent_states=hidden_states,
                return_hidden=True, point_idx=-1)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        final_lines = []
        for subj_idx, line_idx in ids_per_subj.items():

            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r') as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logging.debug("Loading subj {} ({})'s tracking mask."
                              .format(subj_idx, subj_id))
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest')
                tracking_mask.move_to(self.device)

            final_lines.extend(propagate_multiple_lines(
                lines[line_idx], update_memory_after_removing_lines,
                get_dirs_at_last_pos, theta=theta,
                step_size=self.model.step_size, verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))

        return final_lines
