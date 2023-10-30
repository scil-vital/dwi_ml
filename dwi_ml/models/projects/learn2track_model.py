# -*- coding: utf-8 -*-
import logging
from typing import Union, List, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import invert_permutation, PackedSequence, pack_sequence

from dwi_ml.data.processing.space.neighborhood import unflatten_neighborhood
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions, normalize_directions, compute_n_previous_dirs
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    convert_dirs_to_class
from dwi_ml.models.embeddings import NoEmbedding, keys_to_embeddings
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections, ModelWithDirectionGetter,
    ModelWithNeighborhood, MainModelOneInput, ModelOneInputWithEmbedding)
from dwi_ml.models.stacked_rnn import StackedRNN

logger = logging.getLogger('model_logger')  # Same logger as Super.


def faster_unpack_sequence(packed_sequence: PackedSequence):
    # To be used with ordered batch
    # From my tests, seems to be ~twice faster than
    # torch.nn.utils.rnn.unpack_sequence

    # Note:
    # len(batch_size) = max number of points.
    # sum(batch_size) = total number of points
    # values in batch_size: ex, [1000, 1000, 3], means that the first
    # 1000 points are separate lines (we understand that we have 1000 lines).
    # Then the next 1000 points also are separated lines. Then last 3 points
    # are separated lines. So out we have 997 lines with 2 points and
    # 3 with 3 points.
    nb_lines = packed_sequence.batch_sizes[0]

    # Indices of points of the first line.
    ind = [-1]
    for nb_pts in packed_sequence.batch_sizes[:-1]:
        ind.append(ind[-1] + nb_pts)
    ind = np.asarray(ind)

    batch = []
    count_nb_lines_this_size = 0
    remaining_batch_sizes = packed_sequence.batch_sizes.detach().clone()
    total_nb_lines_this_size = remaining_batch_sizes[-1]
    for i in range(nb_lines):
        count_nb_lines_this_size += 1

        if count_nb_lines_this_size > total_nb_lines_this_size:
            # Done for lines of this size. Next size:
            previous_nb = remaining_batch_sizes[-1]
            while remaining_batch_sizes[-1] == previous_nb:
                ind = ind[:-1]
                remaining_batch_sizes = remaining_batch_sizes[:-1]

            count_nb_lines_this_size = 1
            total_nb_lines_this_size = remaining_batch_sizes[-1] - previous_nb

        ind += 1
        line = packed_sequence.data[ind]
        # Note. len(line) == len(remaining_batch_sizes)
        batch.append(line)

    return batch


class Learn2TrackModel(ModelWithPreviousDirections, ModelWithDirectionGetter,
                       ModelWithNeighborhood, ModelOneInputWithEmbedding):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, an RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name,
                 step_size: Union[float, None], compress_lines: Union[float, None],
                 nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedded_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 input_embedding_key: str, input_embedded_size: Union[int, None],
                 nb_cnn_filters: Optional[List[int]],
                 kernel_size: Optional[List[int]],
                 # RNN
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool, use_layer_normalization: bool,
                 dropout: float, start_from_copy_prev: bool,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 # Other
                 neighborhood_type: Optional[str] = None,
                 neighborhood_radius: Optional[int] = None,
                 neighborhood_resolution: Optional[float] = None,
                 log_level=logging.root.level):
        """
        Params
        ------
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        rnn_key: str
            Either 'LSTM' or 'GRU'.
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will depend
            on the skip_connection parameter.
        use_skip_connection: bool
            Whether to use skip connections. See [1] (Figure 1) to visualize
            the architecture.
        use_layer_normalization: bool
            Whether to apply layer normalization to the forward connections.
            See [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        start_from_copy_prev: bool
            If true, final_output = previous_dir + model_output. This can be
            used independantly from the other previous dirs options that define
            values to be concatenated to the input.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        super().__init__(
            experiment_name=experiment_name, step_size=step_size,
            compress_lines=compress_lines, log_level=log_level,
            # For modelWithNeighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,
            # For super ModelWithInputEmbedding:
            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters, kernel_size=kernel_size,
            # For super MainModelWithPD:
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedded_size=prev_dirs_embedded_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # For super ModelForTracking:
            dg_args=dg_args, dg_key=dg_key)

        self.dropout = dropout
        self.start_from_copy_prev = start_from_copy_prev
        self.nb_cnn_filters = nb_cnn_filters
        self.kernel_size = kernel_size

        # Right now input is always flattened (interpolation is implemented
        # that way). For CNN, we will rearrange it ourselves.
        self.input_size = nb_features * self.nb_neighbors

        # ----------- Checks
        if dropout < 0 or dropout > 1:
            raise ValueError('The dropout rate must be between 0 and 1.')

        if start_from_copy_prev and 'gaussian' in dg_key:
            raise ValueError("Start_from_copy_prev makes no sense with "
                             "Gaussian direction getters.")
        if start_from_copy_prev and 'fisher' in dg_key:
            raise ValueError("Start_from_copy_prev makes no sense with "
                             "Fisher von Mises direction getters.")

        # ---------- Instantiations
        # 1. Previous dirs embedding: prepared by super.

        # 2. Input embedding
        self.embedding_dropout = torch.nn.Dropout(self.dropout)

        # 3. Stacked RNN
        if self.nb_previous_dirs > 0:
            self.computed_input_embedded_size += self.prev_dirs_embedded_size
        if len(rnn_layer_sizes) == 1:
            dropout = 0.0  # Not used in RNN. Avoiding the warning.
        self.rnn_model = StackedRNN(
            rnn_key, self.computed_input_embedded_size, rnn_layer_sizes,
            use_skip_connection=use_skip_connection,
            use_layer_normalization=use_layer_normalization, dropout=dropout)

        # 4. Direction getter:
        self.instantiate_direction_getter(self.rnn_model.output_size)

        # If multiple inheritance goes well, these params should be set
        # correctly
        if nb_previous_dirs > 0:
            assert self.forward_uses_streamlines
        assert self.loss_uses_streamlines

        if self.start_from_copy_prev:
            self.forward_uses_streamlines = True

    def set_context(self, context):
        assert context in ['training', 'validation', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        # Every parameter necessary to build the different layers again.
        # during checkpoint state saving.
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'rnn_key': self.rnn_model.rnn_torch_key,
            'rnn_layer_sizes': self.rnn_model.layer_sizes,
            'use_skip_connection': self.rnn_model.use_skip_connection,
            'use_layer_normalization': self.rnn_model.use_layer_normalization,
            'start_from_copy_prev': self.start_from_copy_prev,
            'dropout': self.dropout,
        })

        return params

    @property
    def computed_params_for_display(self):
        p = super().computed_params_for_display
        p['stacked_RNN_output_size'] = self.rnn_model.output_size
        return p

    def forward(self, x: List[torch.tensor],
                input_streamlines: List[torch.tensor] = None,
                hidden_recurrent_states: tuple = None, return_hidden=False,
                point_idx: int = None):
        """Run the model on a batch of sequences.

        Parameters
        ----------
        x: List[torch.tensor]
            Batch of input sequences, i.e. MRI data. Length of the list is the
            number of streamlines in the batch. Each tensor is of size
            [nb_points, nb_features]. During training, should be the length of
            the streamlines minus one (last point is not used). During
            tracking, nb_points should be one; the current point.
        input_streamlines: List[torch.tensor],
            Batch of streamlines. Only used if previous directions are added to
            the model. Used to compute directions; its last point will not be
            used.
        hidden_recurrent_states : tuple
            The current hidden states of the (stacked) RNN model.
        return_hidden: bool
        point_idx: int

        Returns
        -------
        model_outputs : List[Tensor]
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
        out_hidden_recurrent_states : list[states]
            One value per layer.
            LSTM: States are tuples; (h_t, C_t)
                Size of tensors are each [1, nb_streamlines, nb_neurons].
            GRU: States are tensors; h_t.
                Size of tensors are [1, nb_streamlines, nb_neurons].
        """
        # Reminder.
        # Correct interpolation and management of points should be done before.
        if self._context is None:
            raise ValueError("Please set context before usage.")

        # Right now input is always flattened (interpolation is implemented
        # that way). For CNN, we will rearrange it ourselves.
        # Verifying the first input
        assert x[0].shape[-1] == self.input_size, \
            "Not the expected input size! Should be {} (i.e. {} features for " \
            "each of the {} neighbors), but got {} (input shape {})." \
            .format(self.input_size, self.nb_features, self.nb_neighbors,
                    x[0].shape[-1], x[0].shape)

        # Making sure we can use default 'enforce_sorted=True' with packed
        # sequences.
        unsorted_indices = None
        if not self._context == 'tracking':
            # Ordering streamlines per length.
            lengths = torch.as_tensor([len(s) for s in x])
            _, sorted_indices = torch.sort(lengths, descending=True)
            unsorted_indices = invert_permutation(sorted_indices)
            x = [x[i] for i in sorted_indices]
            if input_streamlines is not None:
                input_streamlines = [input_streamlines[i] for i in sorted_indices]

        # ==== 0. Previous dirs.
        n_prev_dirs = None
        copy_prev_dir = 0.0
        if self.nb_previous_dirs > 0 or self.start_from_copy_prev:
            dirs = compute_directions(input_streamlines)
            if self.normalize_prev_dirs:
                dirs = normalize_directions(dirs)

            # Formatting the n previous dirs for last point or all

            # ==== 1. Previous dirs embedding ====
            if self.nb_previous_dirs > 0:
                n_prev_dirs = compute_n_previous_dirs(
                    dirs, self.nb_previous_dirs, point_idx=point_idx)
                n_prev_dirs = pack_sequence(n_prev_dirs)
                # Shape: (nb_points - 1) per streamline x (3 per prev dir)
                n_prev_dirs = self.prev_dirs_embedding(n_prev_dirs.data)
                n_prev_dirs = self.embedding_dropout(n_prev_dirs)

            # Start from copy prev option.
            if self.start_from_copy_prev:
                copy_prev_dir = self.copy_prev_dir(dirs)

        # ==== 2. Inputs embedding ====
        x = pack_sequence(x)
        batch_sizes = x.batch_sizes

        # Avoiding unpacking and packing back if not needed.
        # Input + prev dir embedding required if it's not NoEmbedding.
        if self.nb_previous_dirs > 0 or not isinstance(
                self.input_embedding_layer, NoEmbedding):
            x = x.data

            # Embedding. Shape of inputs: nb_pts_total * embedded_size
            if self.input_embedding_key == 'cnn_embedding':
                # We need to reshape flattened inputs into a neighborhood.
                # Batch has been prepared in self.prepare_batch_one_input.
                x = unflatten_neighborhood(
                    x, self.neighborhood_vectors, self.neighborhood_type,
                    self.neighborhood_radius, self.neighborhood_resolution)

            x = self.input_embedding_layer(x)
            x = self.embedding_dropout(x)

            # ==== 3. Concat with previous dirs ====
            if self.nb_previous_dirs > 0:
                x = torch.cat((x, n_prev_dirs), dim=-1)

            # Shaping again as packed sequence.
            # Shape of inputs.data: nb_pts_total * embedding_size_total
            x = PackedSequence(x, batch_sizes)

        # ==== 3. Stacked RNN (on packed sequence, returns a tensor) ====
        # rnn_output shape: nb_pts_total * last_hidden_layer_size
        assert x.data.shape[-1] == self.rnn_model.input_size, \
            "Expecting input to RNN layer to be of size {}. Got {}" \
            .format(self.rnn_model.input_size, x.data.shape[-1])
        x, out_hidden_recurrent_states = self.rnn_model(
            x, hidden_recurrent_states)

        logger.debug("*** 5. Direction getter....")
        # direction getter can't get a list of sequences.
        # output will be a tensor, but with same format as input.data.
        # we will get a direction for each point.
        assert x.data.shape[-1] == self.direction_getter.input_size, \
            "Expecting input to direction getter to be of size {}. Got {}" \
            .format(self.direction_getter.input_size, x.data.shape[-1])
        x = self.direction_getter(x)

        # Adding either prev_dir or 0.
        if self.start_from_copy_prev:
            x = x + copy_prev_dir

        # Unpacking.
        if not self._context == 'tracking':
            # (during tracking: keeping as one single tensor.)
            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                # Separating mean, sigmas (gaussian) or mean, kappa (fisher)
                x, x2 = x

                x2 = PackedSequence(x2, batch_sizes)
                x2 = faster_unpack_sequence(x2)
                x2 = [x2[i] for i in unsorted_indices]
            x = PackedSequence(x, batch_sizes)
            x = faster_unpack_sequence(x)
            x = [x[i] for i in unsorted_indices]

            if 'gaussian' in self.dg_key or 'fisher' in self.dg_key:
                x = (x, x2)

        if return_hidden:
            # Return the hidden states too. Necessary for the generative
            # (tracking) part, done step by step.
            if not self._context == 'tracking':
                # (ex: when preparing backward tracking.
                #  Must also re-sort hidden states.)
                if self.rnn_model.rnn_torch_key == 'lstm':
                    # LSTM: For each layer, states are tuples; (h_t, C_t)
                    out_hidden_recurrent_states = [
                        (layer_states[0][:, unsorted_indices, :],
                         layer_states[1][:, unsorted_indices, :]) for
                        layer_states in out_hidden_recurrent_states]
                else:
                    # GRU: For each layer, states are tensors; h_t.
                    out_hidden_recurrent_states = [
                        layer_states[:, unsorted_indices, :] for
                        layer_states in out_hidden_recurrent_states]
            return x, out_hidden_recurrent_states
        else:
            return x

    def copy_prev_dir(self, dirs):
        if 'regression' in self.dg_key:
            # Regression: The latest previous dir will be used as skip
            # connection on the output.
            # Either take dirs and add [0, 0, 0] at each first position.
            # Or use pre-computed:
            copy_prev_dir = [torch.nn.functional.pad(cp, [0, 0, 1, 0])
                             for cp in dirs]
            copy_prev_dir = pack_sequence(copy_prev_dir)
            copy_prev_dir = copy_prev_dir.data
        elif self.dg_key == 'sphere-classification':
            # Converting the input directions into classes the same way as
            # during loss, but convert to one-hot.
            # The first previous dir (0) converts to index 0.
            if self._context == 'tracking':
                if dirs[0].shape[0] == 0:
                    copy_prev_dir = torch.zeros(
                        len(dirs),
                        len(self.direction_getter.torch_sphere.vertices),
                        device=self.device)
                else:
                    # Take only the last point.
                    dirs = [d[-1, :][None, :] for d in dirs]
                    copy_prev_dir = convert_dirs_to_class(
                        dirs, self.direction_getter.torch_sphere,
                        smooth_labels=False, add_sos=False, add_eos=False,
                        to_one_hot=True)
                    copy_prev_dir = pack_sequence(copy_prev_dir)
            else:
                # Take all points.
                copy_prev_dir = convert_dirs_to_class(
                    dirs, self.direction_getter.torch_sphere,
                    smooth_labels=False, add_sos=False, add_eos=False,
                    to_one_hot=True)

                # Add zeros as previous dir at the first position
                copy_prev_dir = [torch.nn.functional.pad(cp, [0, 0, 1, 0])
                                 for cp in copy_prev_dir]
                copy_prev_dir = pack_sequence(copy_prev_dir)

            # Making the one from one-hot important for the sigmoid.
            copy_prev_dir = copy_prev_dir.data * 6.0

        elif self.dg_key == 'smooth-sphere-classification':
            raise NotImplementedError
        elif 'gaussian' in self.dg_key:
            # The mean of the gaussian = the previous dir
            raise NotImplementedError
        else:
            # Fisher: not sure how to do that.
            raise NotImplementedError

        return copy_prev_dir

    def remove_lines_in_hidden_state(
            self, hidden_recurrent_states, lines_to_keep):
        """
        Utilitary method to remove a few streamlines from the hidden
        state.
        """
        if self.rnn_model.rnn_torch_key == 'lstm':
            # LSTM: For each layer, states are tuples; (h_t, C_t)
            # Size of tensors are each [1, nb_streamlines, nb_neurons]
            hidden_recurrent_states = [
                (layer_states[0][:, lines_to_keep, :],
                 layer_states[1][:, lines_to_keep, :]) for
                layer_states in hidden_recurrent_states]
        else:
            # GRU: For each layer, states are tensors; h_t.
            # Size of tensors are [1, nb_streamlines, nb_neurons].
            hidden_recurrent_states = [
                layer_states[:, lines_to_keep, :] for
                layer_states in hidden_recurrent_states]
        return hidden_recurrent_states
