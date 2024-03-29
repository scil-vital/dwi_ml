# -*- coding: utf-8 -*-
import logging
from typing import List, Tuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import PackedSequence

keys_to_rnn_class = {'lstm': torch.nn.LSTM,
                     'gru': torch.nn.GRU}

# Note. This logger's logging level can be modified through the main model,
# Learn2trackModel.
logger = logging.getLogger('model_logger')  # Same logger as main dwi_ml.

# Skip connection: In https://arxiv.org/pdf/1308.0850v5.pdf they don't add a
# skip connection to the output. We want to test differently. Ex: With
# previous dirs: should at least learn to reproduce the last dir, and then
# improve from there.
ADD_SKIP_TO_OUTPUT = True


class StackedRNN(torch.nn.Module):
    """
    Recurrent model with recurrent layer sizes, and optional skip connections.

    Needed because Pytorch does not provide a variable layer RNN, nor skip
    connections.
    """

    def __init__(self, rnn_torch_key: str, input_size: int,
                 layer_sizes: List[int], use_skip_connection: bool,
                 use_layer_normalization: bool, dropout: float):
        """
        Parameters
        ----------
        rnn_torch_key : str
            Pytorch class of RNN to instantiate at each layer. Choices are
            'lstm' or 'gru'.
        input_size : int
            Size of each step of the input to the model, i.e. the number of
            features at each step. Note that the complete input will be of
            shape (batch, seq, input_size).
        layer_sizes : list of int
            Size of each hidden layer. The real size will depend
            on the skip_connection parameter.
        use_skip_connection : bool, optional
            If true, concatenate the model input to the input of each hidden
            layer, and concatenate all hidden layers output as the output of
            the model. See [1] (Figure 1) to visualize the architecture.
        use_layer_normalization : bool, optional
            If true, apply layer normalization to the forward connections. See
            [2].
        dropout : float
            If non-zero, introduces a `Dropout` layer on the outputs of each
            RNN layer except the last layer, with given dropout probability.
        ---
        [1] https://arxiv.org/pdf/1308.0850v5.pdf
        [2] https://arxiv.org/pdf/1607.06450.pdf
        """
        if not isinstance(dropout, float) or not 0 <= dropout <= 1:
            raise ValueError("dropout should be a rate in range [0, 1] "
                             "representing the probability of an element "
                             "being zeroed")
        if dropout > 0 and len(layer_sizes) == 1:
            logging.warning("dropout option adds dropout after all but last "
                            "recurrent layer, so non-zero dropout expects "
                            "num_layers greater than 1, but got dropout={} "
                            "and  len(layer_sizes)={}"
                            .format(dropout, len(layer_sizes)))
        if (use_skip_connection and len(layer_sizes) == 1 and not
                ADD_SKIP_TO_OUTPUT):
            logging.warning("With only one layer, the skip connection has "
                            "no effect with current architecture.")
        super().__init__()

        self.rnn_torch_key = rnn_torch_key
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.use_skip_connection = use_skip_connection
        self.use_layer_normalization = use_layer_normalization
        self.dropout = dropout

        self.rnn_layers = []
        self.layer_norm_layers = []
        if self.dropout and self.dropout != 0:
            self.dropout_module = torch.nn.Dropout(self.dropout)
        else:
            self.dropout_module = None
        self.relu_sublayer = torch.nn.ReLU()

        # Initialize model
        rnn_cls = keys_to_rnn_class[self.rnn_torch_key]
        last_layer_size = input_size
        for i, layer_size in enumerate(layer_sizes):
            # Instantiate RNN layer
            # batch_first: If True, then the input and output tensors are
            # provided as (batch, seq, feature), not (seq, batch, feature)
            rnn_layer = rnn_cls(input_size=last_layer_size, hidden_size=layer_size,
                                num_layers=1, batch_first=True)

            # Explicitly add module because it's not a named variable
            self.add_module("rnn_{}".format(i), rnn_layer)
            self.rnn_layers.append(rnn_layer)

            if self.use_layer_normalization:
                layer_norm = torch.nn.LayerNorm(layer_size)
                self.add_module("layer_norm_{}".format(i), layer_norm)
                self.layer_norm_layers.append(layer_norm)

            last_layer_size = layer_size
            # Account for skip connections in layer size. Last layer is
            # different, see self.output_size().
            if self.use_skip_connection:
                last_layer_size += self.input_size

    @property
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        params = {
            'rnn_torch_key': self.rnn_torch_key,
            'input_size': self.input_size,
            'output_size': self.output_size,
            'layer_sizes': list(self.layer_sizes),
            'use_skip_connections': self.use_skip_connection,
            'use_layer_normalization': self.use_layer_normalization,
            'dropout': self.dropout,
        }
        return params

    @property
    def output_size(self):
        """Returns the size of the last layer. If using skip connections, it is
        the sum of all layers' sizes."""
        if self.use_skip_connection:
            if ADD_SKIP_TO_OUTPUT:
                return sum(self.layer_sizes) + self.input_size
            else:
                return sum(self.layer_sizes)
        else:
            return self.layer_sizes[-1]

    def forward(self, inputs: PackedSequence,
                hidden_states: Tuple[Tensor, ...] = None):
        """
        Parameters
        ----------
        inputs : PackedSequence
            Current implementation of the learn2track model calls this using
            packed sequence. We run the RNN on the packed data, but the
            normalization and dropout on their tensor version.
        hidden_states : list[states]
            One value per layer.
            LSTM: States are tuples; (h_t, C_t)
                Size of tensors are each [1, nb_streamlines, nb_neurons].
            GRU: States are tensors; h_t.
                Size of tensors are [1, nb_streamlines, nb_neurons].

        Returns
        -------
        last_output : Tensor
            The PackedSequence.data output, modified through the RNN.
            * You can get the packed results:
            last_output = PackedSequence(last_output,
                                         inputs.batch_sizes,
                                         inputs.sorted_indices,
                                         inputs.unsorted_indices)
            But this can't be used in the direction getter for the next step.
            In our case, skipping.
        out_hidden_states : tuple of Tensor
            The last step hidden states (h_(t-1), C_(t-1) for LSTM) for each
            layer.
        """
        # If input is a tensor: RNN simply runs on it.
        # Else: RNN knows what to do.

        # We need to concatenate initial inputs with skip connections.
        init_inputs = inputs.data

        # Arranging states
        if hidden_states is None:
            hidden_states = [None for _ in range(len(self.rnn_layers))]

        # Initializing variables that we will want to return
        out_hidden_states = []

        # If skip connection, we need to keep in memory the output of all
        # layers
        outputs = []

        # Running forward on each layer:
        # linear --> layer norm --> dropout --> skip connection
        last_output = inputs
        for i in range(len(self.rnn_layers)):
            logger.debug('Applying StackedRnn layer #{}. Layer is: {}'
                         .format(i, self.rnn_layers[i]))

            if i > 0:
                # Packing back the output tensor from previous layer;
                # only the .data was kept from Dropout and Relu
                last_output = PackedSequence(last_output, inputs.batch_sizes)

            # ** RNN **
            # Either as 3D tensor or as packedSequence
            last_output, new_state_i = self.rnn_layers[i](
                last_output, hidden_states[i])
            out_hidden_states.append(new_state_i)

            # ** Other sub-layers **
            # Forward functions for layer_norm, dropout and skip take tensors
            # Does not matter if order of datapoints is not kept, applied on
            # each data point separately
            last_output = last_output.data

            # Apply layer normalization
            if self.use_layer_normalization:
                last_output = self.layer_norm_layers[i](last_output)

            if i < len(self.rnn_layers) - 1:
                # Apply dropout except on last layer
                if self.dropout > 0:
                    last_output = self.dropout_module(last_output)
                    logger.debug('   Output size after dropout: {}'
                                 .format(last_output.shape))

                # Apply ReLu activation except on last layer
                last_output = self.relu_sublayer(last_output)
                logger.debug('   Output size after reLu: {}'
                             .format(last_output.shape))

            # Saving layer's last_output and states for later
            if self.use_skip_connection:
                # Keeping memory for the last layer's concatenation of all
                # outputs.
                outputs.append(last_output)

                # Intermediate layers:
                # Adding skip connection, i.e. initial input.
                # See here: https://arxiv.org/pdf/1308.0850v5.pdf
                if i < len(self.rnn_layers) - 1:
                    last_output = torch.cat((last_output, init_inputs), dim=-1)
                    logger.debug('   Output size after skip connection: {}'
                                 .format(last_output.shape))

        # Final last_output
        if self.use_skip_connection:
            if len(self.rnn_layers) > 1:
                last_output = torch.cat(outputs, dim=-1)
            else:
                last_output = outputs[0]

            if ADD_SKIP_TO_OUTPUT:
                last_output = torch.cat((last_output, init_inputs), dim=-1)
                logger.debug(
                    'Final skip connection: concatenating all outputs AND '
                    'input. Final shape is {}'.format(last_output.shape))
            else:
                logger.debug(
                    'Final skip connection: concatenating all outputs but NOT '
                    'input. Final shape is {}'.format(last_output.shape))
        return last_output, out_hidden_states
