# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.tracking.io_utils import prepare_tracking_mask
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.training.trainers_withGV import \
    DWIMLTrainerForTrackingOneInput

logger = logging.getLogger('trainer_logger')


class Learn2TrackTrainer(DWIMLTrainerForTrackingOneInput):
    """
    Trainer for Learn2Track. Nearly the same as in parent class, but the
    generation-validation phase (tracking) uses the hidden states.
    """
    model: Learn2TrackModel

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def propagate_multiple_lines(self, lines: List[torch.Tensor], ids_per_subj):
        """
        Tractography propagation of 'lines'.
        As compared to super, model requires an additional hidden state.
        """
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        # Setting our own limits here.
        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        # These methods will be used during the loop on subjects
        # Based on the tracker
        def update_memory_after_removing_lines(can_continue: np.ndarray, _):
            # Removing these lines from the hidden state
            nonlocal subjs_hidden_states
            nonlocal subj_idx_in_batch
            subjs_hidden_states[subj_idx_in_batch] = (
                self.model.take_lines_in_hidden_state(
                    subjs_hidden_states[subj_idx_in_batch], can_continue))

        def get_dirs_at_last_pos(subj_lines: List[torch.Tensor], n_last_pos):
            # Get dirs for current subject: run model
            nonlocal subjs_hidden_states
            nonlocal subj_idx
            nonlocal subj_idx_in_batch

            n_last_pos = [pos[None, :] for pos in n_last_pos]
            subj_dict = {subj_idx: slice(0, len(n_last_pos))}
            subj_inputs = self.batch_loader.load_batch_inputs(
                n_last_pos, subj_dict)

            model_outputs, subjs_hidden_states[subj_idx_in_batch] = self.model(
                subj_inputs, subj_lines,
                hidden_recurrent_states=subjs_hidden_states[subj_idx_in_batch],
                return_hidden=True, point_idx=-1)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        # Running the beginning of the streamlines to get the hidden states
        # (using one less point. The next will be done during propagation).
        # Here, subjs_hidden_states will be a list of hidden_states per subj.
        if self.tracking_phase_nb_segments_init > 0:
            tmp_lines = [line[:-1, :] for line in lines]
            inputs = self.batch_loader.load_batch_inputs(
                tmp_lines, ids_per_subj)
            _, whole_hidden_states = self.model(inputs, tmp_lines,
                                                return_hidden=True)

            subjs_hidden_states = [
                self.model.take_lines_in_hidden_state(whole_hidden_states,
                                                      subj_slice)
                for subj, subj_slice in ids_per_subj.items()]
            del tmp_lines, inputs, whole_hidden_states
        else:
            subjs_hidden_states = None

        # Running the propagation separately for each subject
        # (because they all need their own tracking mask)
        final_lines = []
        subj_idx_in_batch = -1
        for subj_idx, subj_line_idx_slice in ids_per_subj.items():
            subj_idx_in_batch += 1   # Will be used as non-local in methods above

            # Load the subject's tracking mask
            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r'
                           ) as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logging.debug("Loading subj {} ({})'s tracking mask."
                              .format(subj_idx, subj_id))
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest')
                tracking_mask.move_to(self.device)

            # Propagates all lines for this subject
            final_lines.extend(propagate_multiple_lines(
                lines[subj_line_idx_slice], update_memory_after_removing_lines,
                get_next_dirs=get_dirs_at_last_pos, theta=theta,
                step_size=self.model.step_size, verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))

        return final_lines
