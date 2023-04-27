# -*- coding: utf-8 -*-
import logging
from typing import Union, List

import numpy as np
import torch
from torch.nn.utils.rnn import PackedSequence, pack_sequence, unpack_sequence

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings as \
    keys_to_tensor_embeddings, NoEmbedding
from dwi_ml.models.main_models import (
    ModelWithPreviousDirections, ModelWithDirectionGetter,
    ModelWithNeighborhood, MainModelOneInput)
from dwi_ml.models.projects.stacked_rnn import StackedRNN

logger = logging.getLogger('model_logger')  # Same logger as Super.


class Learn2TrackModel(ModelWithPreviousDirections, ModelWithDirectionGetter,
                       ModelWithNeighborhood, MainModelOneInput):
    """
    Recurrent tracking model.

    Composed of an embedding for the imaging data's input + for the previous
    direction's input, an RNN model to process the sequences, and a direction
    getter model to convert the RNN outputs to the right structure, e.g.
    deterministic (3D vectors) or probabilistic (based on probability
    distribution parameters).
    """

    def __init__(self, experiment_name,
                 step_size: Union[float, None], compress: Union[float, None],
                 nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedding_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 input_embedding_key: str,
                 input_embedding_size: Union[int, None],
                 input_embedding_size_ratio: Union[float, None],
                 # RNN
                 rnn_key: str, rnn_layer_sizes: List[int],
                 use_skip_connection: bool,
                 use_layer_normalization: bool, dropout: float,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Params
        ------
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        input_embedding_key: str
            Key to an embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings).
            Default: 'no_embedding'.
        input_embedding_size: int
            Output embedding size for the input. If None, will be set to
            input_size.
        input_embedding_size_ratio: float
            Other possibility to define input_embedding_size, which then equals
            [ratio * (nb_features * (nb_neighbors+1))]
        rnn_key: str
            Either 'LSTM' or 'GRU'.
        rnn_layer_sizes: List[int]
            The list of layer sizes for the rnn. The real size will depend
            on the skip_connection parameter.
        use_skip_connection: bool
            Whether to use skip connections. See [1] (Figure 1) to visualize
            the architecture.
        use_layer_normalization: bool
            Wheter to apply layer normalization to the forward connections. See
            [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        if prev_dirs_embedding_key == 'no_embedding':
            if prev_dirs_embedding_size is None:
                prev_dirs_embedding_size = 3 * nb_previous_dirs
            elif prev_dirs_embedding_size != 3 * nb_previous_dirs:
                raise ValueError("To use identity embedding, the output size "
                                 "must be the same as the input size!")

        super().__init__(
            experiment_name=experiment_name, step_size=step_size,
            compress=compress, log_level=log_level,
            # For modelWithNeighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            # For super MainModelWithPD:
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedding_size=prev_dirs_embedding_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # For super ModelForTracking:
            dg_args=dg_args, dg_key=dg_key)

        self.input_embedding_key = input_embedding_key
        self.nb_features = nb_features
        self.dropout = dropout

        # ----------- Checks
        if self.input_embedding_key not in keys_to_tensor_embeddings.keys():
            raise ValueError("Embedding choice for x data not understood: {}"
                             .format(self.embedding_key_x))

        # ---------- Instantiations
        # 1. Previous dirs embedding: prepared by super.

        # 2. Input embedding
        self.input_size = nb_features * (self.nb_neighbors + 1)
        if input_embedding_size and input_embedding_size_ratio:
            raise ValueError("You must only give one value, either "
                             "input_embedding_size or "
                             "input_embedding_size_ratio")
        elif input_embedding_size_ratio:
            self.input_embedding_size = int(self.input_size *
                                            input_embedding_size_ratio)
        elif input_embedding_size:
            self.input_embedding_size = input_embedding_size
        else:
            self.input_embedding_size = self.input_size
        self.embedding_dropout = torch.nn.Dropout(self.dropout)

        input_embedding_cls = keys_to_tensor_embeddings[input_embedding_key]
        self.input_embedding = input_embedding_cls(
            input_size=self.input_size, output_size=self.input_embedding_size)

        # 3. Stacked RNN
        rnn_input_size = self.input_embedding_size
        if self.nb_previous_dirs > 0:
            rnn_input_size += self.prev_dirs_embedding_size
        if len(rnn_layer_sizes) == 1:
            dropout = 0.0  # Not used in RNN. Avoiding the warning.
        self.rnn_model = StackedRNN(
            rnn_key, rnn_input_size, rnn_layer_sizes,
            use_skip_connection=use_skip_connection,
            use_layer_normalization=use_layer_normalization,
            dropout=dropout)

        # 4. Direction getter:
        self.instantiate_direction_getter(self.rnn_model.output_size)

        # If multiple inheritance goes well, these params should be set
        # correctly
        if nb_previous_dirs > 0:
            assert self.forward_uses_streamlines
        assert self.loss_uses_streamlines

    def set_context(self, context):
        assert context in ['training', 'tracking', 'visu',
                           'preparing_backward']
        self._context = context

    @property
    def params_for_checkpoint(self):
        # Every parameter necessary to build the different layers again.
        # during checkpoint state saving.
        params = super().params_for_checkpoint
        params.update({
            'nb_features': int(self.nb_features),
            'input_embedding_key': self.input_embedding_key,
            'input_embedding_size': int(self.input_embedding_size),
            'input_embedding_size_ratio': None,
            'rnn_key': self.rnn_model.rnn_torch_key,
            'rnn_layer_sizes': self.rnn_model.layer_sizes,
            'use_skip_connection': self.rnn_model.use_skip_connection,
            'use_layer_normalization': self.rnn_model.use_layer_normalization,
            'dropout': self.dropout,
        })

        return params

    def forward(self, inputs: List[torch.tensor],
                target_streamlines: List[torch.tensor] = None,
                hidden_recurrent_states: tuple = None):
        """Run the model on a batch of sequences.

        Parameters
        ----------
        inputs: List[torch.tensor]
            Batch of input sequences, i.e. MRI data. Length of the list is the
            number of streamlines in the batch. Each tensor is of size
            [nb_points, nb_features]. During training, should be the length of
            the streamlines minus one (last point is not used). During
            tracking, nb_points should be one; the current point.
        target_streamlines: List[torch.tensor],
            Batch of streamlines. Only used if previous directions are added to
            the model. Used to compute directions; its last point will not be
            used.
        hidden_recurrent_states : tuple
            The current hidden states of the (stacked) RNN model.

        Returns
        -------
        model_outputs : Tensor
            Output data, ready to be passed to either `compute_loss()` or
            `get_tracking_directions()`.
            NOTE: this tensor's format will be one direction per point in the
            input, with the same organization os the initial packed sequence.
            It should be compared with packed_sequences's .data.
            Or it is possible to form pack a packed sequence with
            output = PackedSequence(output,
                                   inputs.batch_sizes,
                                   inputs.sorted_indices,
                                   inputs.unsorted_indices)
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

        # Ordering of PackedSequence for 1) inputs, 2) previous dirs and
        # 3) targets (when computing loss) may not always be the same.
        # Pack inputs now and use that information for others.
        # Shape of inputs.data: nb_pts_total * nb_features
        if self._context == 'tracking':
            # We should have only one input per streamline (at the last
            # coordinate).
            assert np.all([len(i) == 1] for i in inputs)

            # Using vstack rather than packing directly, to supervise their
            # order.
            nb_streamlines = len(inputs)
            inputs = torch.vstack(inputs)
            batch_sizes = torch.as_tensor([nb_streamlines])
            sorted_indices = torch.arange(nb_streamlines, device=self.device)
            unsorted_indices = torch.arange(nb_streamlines, device=self.device)
            inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                    unsorted_indices)
        else:
            # In all other cases, len(each input) == len(each streamline),
            # if any (only used with previous dirs).
            if target_streamlines is not None:
                assert np.all([len(i) == len(s) for i, s in
                               zip(inputs, target_streamlines)]), \
                    "Expecting same nb of inputs and streamlines... " \
                    "Got {} vs {} \n(p.s. context is: {})" \
                    .format([len(i) for i in inputs],
                            [len(s) for s in target_streamlines],
                            self._context)

            # Streamlines have different lengths. Packing.
            inputs = pack_sequence(inputs, enforce_sorted=False)
            batch_sizes = inputs.batch_sizes
            sorted_indices = inputs.sorted_indices
            unsorted_indices = inputs.unsorted_indices

        # ==== 1. Previous dirs embedding ====
        if self.nb_previous_dirs > 0:
            prev_dirs = compute_directions(target_streamlines)
            point_idx = -1 if self._context == 'tracking' else None

            # Result will be a packed sequence.
            n_prev_dirs_embedded = self.normalize_and_embed_previous_dirs(
                prev_dirs, unpack_results=False, point_idx=point_idx,
                sorted_indices=sorted_indices)
            n_prev_dirs_embedded = self.embedding_dropout(
                n_prev_dirs_embedded.data)
        else:
            n_prev_dirs_embedded = None

        # ==== 2. Inputs embedding ====
        # Avoiding unpacking and packing back if not needed.
        if self.nb_previous_dirs > 0 or not isinstance(
                self.input_embedding, NoEmbedding):

            # Embedding. Shape of inputs: nb_pts_total * embedding_size
            inputs = self.input_embedding(inputs.data)
            inputs = self.embedding_dropout(inputs)

            # ==== 3. Concat with previous dirs ====
            if n_prev_dirs_embedded is not None:
                inputs = torch.cat((inputs, n_prev_dirs_embedded), dim=-1)

            # Shaping again as packed sequence.
            # Shape of inputs.data: nb_pts_total * embedding_size_total
            inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                    unsorted_indices)

        # ==== 3. Stacked RNN (on packed sequence, returns a tensor) ====
        # rnn_output shape: nb_pts_total * last_hidden_layer_size
        rnn_output, out_hidden_recurrent_states = self.rnn_model(
            inputs, hidden_recurrent_states)

        logger.debug("*** 5. Direction getter....")
        # direction getter can't get a list of sequences.
        # output will be a tensor, but with same format as input.data.
        # we will get a direction for each point.
        model_outputs = self.direction_getter(rnn_output)

        # Return the hidden states. Necessary for the generative
        # (tracking) part, done step by step.

        # During training / visu: not unpacking now; we will compute the loss
        # point by whole tensor.
        # But sending as PackedSequence to be sure that targets
        # will be concatenated in the same order when computing loss.
        # During tracking last point: keeping as one tensor.
        # During tracking backward: ignoring output anyway. Only computing
        # hidden state.
        if not self._context == 'tracking':
            model_outputs = PackedSequence(model_outputs, batch_sizes,
                                           sorted_indices, unsorted_indices)

        if self._context == 'tracking':
            # Return the hidden states too.
            return model_outputs, out_hidden_recurrent_states
        else:
            return model_outputs

    def compute_loss(self, model_outputs: PackedSequence,
                     target_streamlines: List[torch.Tensor],
                     average_results=True, **kw):
        # Prepare targets to the correct format.
        target_dirs = self.direction_getter.prepare_targets_for_loss(
            target_streamlines)

        # Packing dirs and using the .data instead of looping on streamlines.
        # Anyway, loss is computed point by point.
        target_dirs = pack_sequence(target_dirs, enforce_sorted=False)
        assert torch.equal(target_dirs.sorted_indices,
                           model_outputs.sorted_indices)

        # Computing loss
        loss = self.direction_getter.compute_loss(
            model_outputs.data, target_dirs.data, average_results)

        if not average_results:
            # Will be easier to manage if streamlines are stacked rather than
            # packed. Will be like other models.
            loss_packed = PackedSequence(
                loss, batch_sizes=model_outputs.batch_sizes,
                sorted_indices=model_outputs.sorted_indices,
                unsorted_indices=model_outputs.unsorted_indices)
            loss = unpack_sequence(loss_packed)
            loss = torch.hstack(loss)
        return loss
