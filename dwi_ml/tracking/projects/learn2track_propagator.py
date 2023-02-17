# -*- coding: utf-8 -*-
import logging

import torch

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.tracking.propagator import (DWIMLPropagatorOneInput,
                                        DWIMLPropagatorwithStreamlineMemory)

logger = logging.getLogger('tracker_logger')


class RecurrentPropagator(DWIMLPropagatorOneInput,
                          DWIMLPropagatorwithStreamlineMemory):
    """
    To use a RNN for a generative process, the hidden recurrent states that
    would be passed (ex, h_(t-1), C_(t-1) for LSTM) need to be kept in memory
    as an additional input.

    In theory, the previous timepoints' inputs do not need to be kept, except
    for the backward tracking: the hidden recurrent states need to be computed
    from scratch. We will reload them all when starting backward, if necessary.
    """
    model: Learn2TrackModel

    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
                 model: Learn2TrackModel, input_volume_group: str,
                 step_size: float, algo: str, theta: float, device=None,
                 normalize_directions: bool = True):
        super().__init__(dataset=dataset,
                         subj_idx=subj_idx, model=model,
                         input_volume_group=input_volume_group,
                         step_size=step_size, algo=algo, theta=theta,
                         device=device, verify_opposite_direction=False,
                         normalize_directions=normalize_directions)

        # Internal state:
        # - previous_dirs, already dealt with by super.
        # - For RNN: new parameter: The hidden states of the RNN
        self.hidden_recurrent_states = None

    def prepare_forward(self, seeding_pos):
        """
        Additionnally to usual preparation, we need to reset the recurrent
        hidden state.
        """
        logger.debug("Learn2track: Resetting propagator for new "
                     "streamline(s).")
        self.hidden_recurrent_states = None

        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, lines, forward_dir):
        """
        Preparing backward. We need to recompute the hidden recurrent state
        for this half-streamline.
        """
        logger.debug("Computing hidden RNN state at backward: run model on "
                     "(reversed) first half.")

        # Must re-run the model from scratch to get the hidden states
        # Either load all timepoints in memory and call model once.
        # Or loop.
        all_inputs, _ = self.model.prepare_batch_one_input(
            lines, self.dataset, self.subj_idx, self.volume_group)

        # all_inputs is a tuple of
        # nb_streamlines x tensor[nb_points, nb_features]

        # Running model. If we send is_tracking=True, will only compute the
        # previous dirs for the last point. To mimic training, we have to
        # add an additional fake point to the streamline, not used (during
        # training, only used to compute the loss).
        lines_with0 = [torch.cat((line, torch.zeros(1, 3, device=self.device)),
                                 dim=0)
                       for line in lines]

        # Also, warning: creating a tensor from a list of np arrays is low.
        _, self.hidden_recurrent_states = self.model(
            all_inputs, lines_with0, is_tracking=False, return_state=True)

        return super().prepare_backward(lines, forward_dir)

    def _call_model_forward(self, inputs, lines):
        # For RNN, we need to send the hidden state too.
        model_outputs, self.hidden_recurrent_states = self.model(
            inputs, lines, self.hidden_recurrent_states,
            return_state=True, is_tracking=True)

        return model_outputs

    def multiple_lines_update(self, lines_that_continue: list):
        """
        Removing rejected lines from hidden states

        Params
        ------
        lines_that_continue: list
            List of indexes of lines that are kept.
        """
        # Hidden states: list[states] (One value per layer).
        nb_streamlines = len(self.current_lines)
        all_idx = torch.zeros(nb_streamlines)
        all_idx[lines_that_continue] = 1

        if self.model.rnn_model.rnn_torch_key == 'lstm':
            # LSTM: States are tuples; (h_t, C_t)
            # Size of tensors are each [1, nb_streamlines, nb_neurons]
            self.hidden_recurrent_states = [
                (hidden_states[0][:, lines_that_continue, :],
                 hidden_states[1][:, lines_that_continue, :]) for
                hidden_states in self.hidden_recurrent_states]
        else:
            #   GRU: States are tensors; h_t.
            #     Size of tensors are [1, nb_streamlines, nb_neurons].
            self.hidden_recurrent_states = [
                hidden_states[:, lines_that_continue, :] for
                hidden_states in self.hidden_recurrent_states]
