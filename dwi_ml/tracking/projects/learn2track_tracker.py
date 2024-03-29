# -*- coding: utf-8 -*-
import logging

import numpy as np

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.tracking.tracker import DWIMLTrackerOneInput

logger = logging.getLogger('tracker_logger')


class RecurrentTracker(DWIMLTrackerOneInput):
    """
    To use an RNN for a generative process, the hidden recurrent states that
    would be passed (ex, h_(t-1), C_(t-1) for LSTM) need to be kept in memory
    as an additional input.

    In theory, the previous timepoints' inputs do not need to be kept, except
    for the backward tracking: the hidden recurrent states need to be computed
    from scratch. We will reload them all when starting backward, if necessary.
    """
    model: Learn2TrackModel

    def __init__(self, **kw):
        super().__init__(verify_opposite_direction=False, **kw)

        # Internal state:
        # - previous_dirs, already dealt with by super.
        # - For RNN: new parameter: The hidden states of the RNN
        self.hidden_recurrent_states = None

    def prepare_forward(self, seeding_pos):
        """
        Additionnally to usual preparation, we need to reset the recurrent
        hidden state.
        """
        self.hidden_recurrent_states = None

        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, lines):
        """
        Preparing backward. We need to recompute the hidden recurrent state
        for this half-streamline.
        """
        lines, rej_idx = super().prepare_backward(lines)

        logger.debug("Computing hidden RNN state at backward: run model on "
                     "(reversed) first half.")

        # Must re-run the model from scratch to get the hidden states
        self.model.set_context('preparing_backward')

        # We don't re-run the last point (i.e. the seed) because the first
        # propagation step after backward = at that point.
        tmp_lines = [s[:-1, :] for s in lines]
        all_inputs = self.model.prepare_batch_one_input(
            tmp_lines, self.dataset, self.subj_idx, self.volume_group)

        # all_inputs is a List of
        # nb_streamlines x tensor[nb_points, nb_features]

        # No hidden state given = running model on all points.
        with self.grad_context:
            _, self.hidden_recurrent_states = self.model(
                all_inputs, tmp_lines, return_hidden=True, point_idx=None)

        # Back to tracking context
        self.model.set_context('tracking')

        return lines, rej_idx

    def _call_model_forward(self, inputs, lines):
        # For RNN, we need to send the hidden state too.
        with self.grad_context:
            model_outputs, self.hidden_recurrent_states = self.model(
                inputs, lines, self.hidden_recurrent_states,
                return_hidden=True, point_idx=-1)

        return model_outputs

    def update_memory_after_removing_lines(self, can_continue: np.ndarray, _):
        """
        Removing rejected lines from hidden states.

        Params
        ------
        can_continue: np.ndarray
            Indexes of lines that are kept.
        """
        # Hidden states: list[states] (One value per layer).
        self.hidden_recurrent_states = self.model.take_lines_in_hidden_state(
            self.hidden_recurrent_states, can_continue)
