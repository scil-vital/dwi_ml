# -*- coding: utf-8 -*-
import logging
from typing import Union, List, Tuple

import numpy as np
import torch
from torch.nn import Dropout, Transformer
from torch.nn.functional import pad

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.main_models import (MainModelOneInput,
                                       ModelForTracking,
                                       ModelWithPreviousDirections,
                                       ModelWithNeighborhood)
from dwi_ml.models.projects.positional_encoding import \
    keys_to_positional_encodings
from dwi_ml.models.utils.transformers_from_torch import (
    ModifiedTransformer,
    ModifiedTransformerEncoder, ModifiedTransformerEncoderLayer,
    ModifiedTransformerDecoder, ModifiedTransformerDecoderLayer)

# Our model needs to be auto-regressive, to allow inference / generation at
# tracking time.
# => During training, we hide the future; both in the input and in the target
# sequences.

# About the tracking process
# At each new step, the whole sequence is processed again (ran in the model).
# We only keep the last output. This is not very efficient... Is there a way
# to keep the hidden state in-between?
logger = logging.getLogger('model_logger')  # Same logger as Super.

# Data dimensions:
# If batch_first=True, then the input and output tensors are provided as
# (batch, seq, feature). If False, (seq, batch, feature).
# Here prepared code with dim (nb streamlines, max_len, d_model), which
# corresponds to True.
BATCH_FIRST = True
SOS_TOKEN = 0.0


def forward_padding(data: torch.tensor, expected_length):
    return pad(data, (0, 0, 0, expected_length - len(data)))


class AbstractTransformerModel(ModelWithPreviousDirections,
                               ModelWithNeighborhood,
                               MainModelOneInput, ModelForTracking):
    """
    Prepares the parts common to our two transformer versions: embeddings,
    direction getter and some parameters for the model.

    Encoder and decoder will be prepared in child classes.

    About data embedding:
    We could use the raw data, technically. But when adding the positional
    embedding, the reason it works is that the learning of the embedding
    happens while knowing that some positional vector will be added to it.
    As stated in the blog
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    the embedding probably adapts to leave place for the positional encoding.
    """
    # mask_sos: bool
    #       The input to the decoder is the shifted target sequence with an
    #       additional Start Of Sequence (SOS) token in the first position.
    #       Considering as we do not work with a dictionary of token, this is
    #       not straightforward here. We use (0,0,..., 0) as SOS value. We can
    #       hope that the model will learn its signification, or hide it in the
    #       attention.
    # Option mask_sos does not work yet. Creates some nan in the attention.
    # We can solve the forward propagation with nantonum (setting to 0)
    # but the backward propagation fails.
    mask_sos = False

    def __init__(self,
                 experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedding_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 max_len: int, positional_encoding_key: str,
                 embedding_key_x: str, embedding_key_t: str,
                 # TRANSFORMER
                 d_model: int, ffnn_hidden_size: Union[int, None],
                 nheads: int, dropout_rate: float, activation: str,
                 norm_first: bool,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 normalize_targets: bool,
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Args
        ----
        experiment_name: str
            Name of the experiment.
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        nb_previous_dirs: int
            Number of previous direction to concatenate to each input.
            Default: 0.
        prev_dirs_embedding_size: int
            Dimension of the final vector representing the previous directions
            (no matter the number of previous directions used).
            Default: nb_previous_dirs * 3.
        prev_dirs_embedding_key: str,
            Key to an embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings).
            Default: None (no previous directions added).
        normalize_prev_dirs: bool
            If true, direction vectors are normalized (norm=1) when computing
            the previous direction.
        positional_encoding_key: str,
            Chosen class for the input's positional embedding. Choices:
            keys_to_positional_embeddings.keys(). Default: 'sinusoidal'.
        embedding_key_x: str,
            Chosen class for the input embedding (the data embedding part).
            Choices: keys_to_embeddings.keys().
            Default: 'no_embedding'.
        embedding_key_t: str,
            Target embedding, with the same choices as above.
            Default: 'no_embedding'.
        d_model: int,
            The transformer REQUIRES the same output dimension for each layer
            everywhere to allow skip connections. = d_model. Note that
            embeddings should also produce outputs of size d_model.
            Value must be divisible by num_heads.
            Default: 4096.
        ffnn_hidden_size: int
            Size of the feed-forward neural network (FFNN) layer in the encoder
            and decoder layers. The FFNN is composed of two linear layers. This
            is the size of the output of the first one. In the music paper,
            = d_model/2. Default: d_model/2.
        nheads: int
            Number of attention heads in each attention or self-attention
            layer. Default: 8.
        dropout_rate: float
            Dropout rate. Constant in every dropout layer. Default: 0.1.
        activation: str
            Choice of activation function in the FFNN. 'relu' or 'gelu'.
            Default: 'relu'.
        norm_first: bool
            If True, encoder and decoder layers will perform LayerNorms before
            other attention and feedforward operations, otherwise after.
            Torch default + in original paper: False. In the tensor2tensor code
            they suggest that learning is more robust when preprocessing each
            layer with the norm.
        dg_key: str
            Key to the chosen direction getter class. Choices:
            keys_to_direction_getters.keys(). Default: 'cosine-regression'.
        dg_args: dict
            Arguments necessary for the instantiation of the chosen direction
            getter.
        normalize_targets: bool
            If true, target streamline's directions vectors are normalized
            (norm=1). If the step size is fixed, it shouldn't make any
            difference. If streamlines are compressed, in theory you should
            normalize, but you could hope that not normalizing could give back
            to the algorithm a sense of distance between points.
            If true and the dg_key is a regression model, then, output
            directions are also normalized too. Default: True.
        neighborhood_type: str
            The type of neighborhood to add. One of 'axes', 'grid' or None. If
            None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). (Can be none)
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
        """
        super().__init__(
            # MainAbstract
            experiment_name=experiment_name, log_level=log_level,
            # PreviousDirs
            nb_previous_dirs=nb_previous_dirs,
            prev_dirs_embedding_size=prev_dirs_embedding_size,
            prev_dirs_embedding_key=prev_dirs_embedding_key,
            normalize_prev_dirs=normalize_prev_dirs,
            # Neighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            # Tracking
            dg_key=dg_key, dg_args=dg_args,
            normalize_targets=normalize_targets)

        self.nb_features = nb_features
        self.max_len = max_len
        self.positional_encoding_key = positional_encoding_key
        self.embedding_key_x = embedding_key_x
        self.embedding_key_t = embedding_key_t
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.nheads = nheads
        self.d_model = d_model
        self.ffnn_hidden_size = ffnn_hidden_size if ffnn_hidden_size \
            else d_model // 2
        self.norm_first = norm_first

        # ----------- Checks
        if self.embedding_key_x not in keys_to_embeddings.keys():
            raise ValueError("Embedding choice for x data not understood: {}"
                             .format(self.embedding_key_x))
        if self.embedding_key_t not in keys_to_embeddings.keys():
            raise ValueError("Embedding choice for targets not understood: {}"
                             .format(self.embedding_key_t))
        if self.positional_encoding_key not in \
                keys_to_positional_encodings.keys():
            raise ValueError("Positional encoding choice not understood: {}"
                             .format(self.positional_encoding_key))
        if self.dg_key not in keys_to_direction_getters.keys():
            raise ValueError("Direction getter choice not understood: {}"
                             .format(self.positional_encoding_key))
        assert d_model // nheads == float(d_model) / nheads, \
            "d_model ({}) must be divisible by nheads ({})".format(d_model,
                                                                   nheads)

        # ----------- Input size:
        # (neighborhood prepared by super)
        self.input_size = nb_features * (self.nb_neighbors + 1)

        # ----------- Instantiations
        # 1. Previous dirs embedding: prepared by super.

        # 2. x embedding
        cls_x = keys_to_embeddings[self.embedding_key_x]
        # output will be concatenated with prev_dir embedding and total must
        # be d_model.
        if self.nb_previous_dirs > 0:
            embed_size = d_model - self.prev_dirs_embedding_size
        else:
            embed_size = d_model
        self.embedding_layer_x = cls_x(self.input_size, embed_size)
        # This dropout is only used in the embedding; torch's transformer
        # prepares its own dropout elsewhere.
        self.dropout = Dropout(self.dropout_rate)

        # 3. positional encoding
        cls_p = keys_to_positional_encodings[self.positional_encoding_key]
        self.position_encoding_layer = cls_p(d_model, dropout_rate, max_len)

        # 4. target embedding
        cls_t = keys_to_embeddings[self.embedding_key_t]
        self.embedding_layer_t = cls_t(3, d_model)

        # 5. Transformer: See child classes

        # 6. Direction getter
        # Original paper: last layer = Linear + Softmax on nb of classes.
        # Note on parameter initialization.
        # They all use torch.nn.linear, which initializes parameters based
        # on a kaiming uniform, same as uniform(-sqrt(k), sqrt(k)) where k is
        # the nb of features.
        self.instantiate_direction_getter(d_model)

    @property
    def params_for_json_prints(self):
        p = super().params_for_json_prints
        p.update(self.params_for_checkpoint)
        return p

    @property
    def params_for_checkpoint(self):
        """
        Every parameter necessary to build the different layers again
        from a checkpoint.
        """
        p = super().params_for_checkpoint
        p.update({
            'nb_features': int(self.nb_features),
            'max_len': self.max_len,
            'embedding_key_x': self.embedding_key_x,
            'positional_encoding_key': self.positional_encoding_key,
            'embedding_key_t': self.embedding_key_t,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'nheads': self.nheads,
            'd_model': self.d_model,
            'ffnn_hidden_size': self.ffnn_hidden_size,
            'norm_first': self.norm_first,
        })
        return p

    def _stack_batch(self, unpadded_data, pad_first: bool, pad_length: int):
        """
        Pad the batch inputs and targets so that all streamlines have lenth
        max_len. Then concatenate all streamlines.

        Params
        ------
        batch_x: list[Tensor]
            Len: nb streamlines. Shape of each tensor: nb points x nb features.
        pad_first: bool
            If false, padding is skipped. (Ex: If all streamlines already
            contain the right number of points.
        pad_length: int
            Expected final lengths of streamlines.

        Returns
        -------
        formatted_x: Tensor
            Shape [nb_streamlines, max_len, nb_features] where nb features is
            the size of the batch input at this point (ex, initial number of
            features or d_model if embedding is already done).
        """
        if pad_first:
            padded_data = [forward_padding(unpadded_data[i], pad_length)
                           for i in range(len(unpadded_data))]
        else:
            padded_data = unpadded_data

        return torch.stack(padded_data).to(self.device)

    @staticmethod
    def _generate_future_mask(sz):
        """DO NOT USE FLOAT, their code had a bug (see issue #92554. Fixed in
        latest github branch. Waiting for release.)

        Float is deprecated but their method has not been changed yet.
        Should be boolean. I warned them, not changed yet.
        """
        mask = Transformer.generate_square_subsequent_mask(sz)
        return mask < 0

    @staticmethod
    def _generate_padding_mask(unpadded_lengths, batch_max_len):
        nb_streamlines = len(unpadded_lengths)

        mask_padding = torch.full((nb_streamlines, batch_max_len),
                                  fill_value=False)
        for i in range(nb_streamlines):
            mask_padding[i, unpadded_lengths[i]:] = True

        return mask_padding

    def _prepare_masks(self, unpadded_lengths, use_padding, batch_max_len):
        """
        Prepare masks for the transformer.

        Params
        ------
        unpadded_lengths: list
            Length of each streamline's target sequence.
            During tracking unpadded lenghts is considered the same for x and
            t. If is_training, input is one point longer.
        use_padding: bool,
            If false, skip padding (all streamlines must have the same length).
        batch_max_len: int
            Batch's maximum length. It it not useful to pad more than that.
            (During tracking, particularly interesting!). Should be equal or
            smaller to self.max_len.

        Returns
        -------
        mask_future_x: Tensor
            Shape: [batch_max_len, batch_max_len]
            Masks the inputs that do not exist (useful for data generation, but
            also used during training on padded points because, why not), at
            each position.
            = [[False, True,  True],
               [False, False, True],
               [False, False, False]]
        mask_future_t: Tensor
            Idem, but we can choose to hide the fake SOS token. If so, the
            first column is set to True.
        mask_padding: Tensor
            Shape [nb_streamlines, batch_max_len]. Masks positions that do not
            exist in the sequence.
        """
        mask_future_x = self._generate_future_mask(batch_max_len)
        if self.mask_sos:
            mask_future_t = self._generate_future_mask(batch_max_len)
            mask_future_t[:, 0] = True
        else:
            mask_future_t = mask_future_x

        if use_padding:
            mask_padding = self._generate_padding_mask(unpadded_lengths,
                                                       batch_max_len)
        else:
            mask_padding = None

        return mask_future_x, mask_future_t, mask_padding

    def forward(self, batch_x: List[torch.tensor],
                batch_streamlines: List[torch.tensor],
                is_tracking=False, return_weights=False,
                average_heads=False, **kw):
        """
        Params
        ------
        batch_x: list[Tensor]
            One tensor per streamline. Size of each tensor =
            [nb_input_points, nb_features].
        batch_streamlines: list[Tensor]
            Streamline coordinates. One tensor per streamline. Size of each
            tensor = [nb_input_points + 1, 3]. Directions will be computed to
            obtain targets of the same lengths. Then, directions are used for
            two things:
            - As input to the decoder. This input is generally the shifted
            sequence, with a SOS token (start of sequence) at the first
            position. In our case, there is no token, but the sequence is
            adequately masked to hide future positions. The last direction is
            not used.
            - As target during training. The whole sequence is used.
        is_tracking: bool
            During tracking, the input sequences contain one more point than
            the directions sequences.
        return_weights: bool
            If true, returns the weights of the attention layers.
        average_heads: bool
            If return_weights, you may choose to average the weights from
            different heads together.

        Returns
        -------
        output: Tensor,
            Batch output, of shape:
                - During training: [total nb points all streamlines, out size]
                - During tracking: [nb streamlines * 1, out size]
        weights: Tuple
            If return weigts: The weights (depending on the child model)
        """
        # Remember lengths to unpad outputs later. During tracking, the
        # targets contain one less point.
        unpadded_lengths = np.asarray([len(i) for i in batch_x])
        nb_streamlines = len(batch_streamlines)

        # ----------- Checks
        if np.any(unpadded_lengths > self.max_len):
            raise ValueError("Some streamlines were longer than accepted max "
                             "length for sequences ({})".format(self.max_len))
        if is_tracking:
            for i in range(nb_streamlines):
                assert len(batch_streamlines[i]) == len(batch_x[i]), \
                    "During tracking, we expect the streamlines to have the " \
                    "same number of points as the input."
        else:
            for i in range(nb_streamlines):
                assert len(batch_streamlines[i]) == len(batch_x[i]) + 1, \
                    "We expect the streamlines to have one more point " \
                    "than the inputs. Directions will be used. No need " \
                    "to compute the last input. (During training)."

        # ----------- Ok. Prepare masks and parameters

        # (Skip padding if all streamlines have the same length)
        use_padding = not np.all(unpadded_lengths == unpadded_lengths[0])
        batch_max_len = np.max(unpadded_lengths)
        masks = self._prepare_masks(
            unpadded_lengths, use_padding, batch_max_len)

        # Compute targets (= directions).
        # Will be computed again later for loss computation, but ok, should not
        # be too heavy.
        batch_t = compute_directions(batch_streamlines)

        # ----------- Ok. Start processing

        # 1. Embedding + position encoding.
        # Run embedding on padded data. Necessary to make the model
        # adapt for the positional encoding.
        embed_x, embed_t = self.run_embedding(batch_x, batch_t, use_padding,
                                              batch_max_len)
        embed_x, embed_t = self.dropout(embed_x), self.dropout(embed_t)

        # 2. Main transformer
        outputs, weights = self._run_main_layer_forward(
            embed_x, embed_t, masks, return_weights, average_heads)

        # Unpad now and combine everything for the direction getter.
        if is_tracking:
            outputs = outputs.detach()
            # No need to actually unpad, we only take the last (unpadded)
            # point, newly created.
            # (len = len(x) so that's the new point. -1 for python indexing)
            outputs = [outputs[i, unpadded_lengths[i] - 1, :]
                       for i in range(nb_streamlines)]
            outputs = torch.vstack(outputs)
        else:
            if use_padding:
                # outputs size = [nb streamlines, max_len, d_model].
                # We take all (unpadded) points.
                outputs = [outputs[i, 0:unpadded_lengths[i], :]
                           for i in range(nb_streamlines)]
            outputs = torch.cat(outputs, dim=0)

        # 3. Direction getter
        # Outputs will be all streamlines merged.
        # To compute loss = ok. During tracking, we will need to split back.
        outputs = self.direction_getter(outputs)

        if return_weights:
            return outputs, weights
        return outputs

    def run_embedding(self, batch_x, batch_t, use_padding, batch_max_len):
        """
        Pad + concatenate.
        Embedding. (Add SOS token to target.)
        Positional encoding.
        """
        # Inputs
        inputs = self._stack_batch(batch_x, use_padding, batch_max_len)
        inputs = self.embedding_layer_x(inputs)
        if self.nb_previous_dirs > 0:
            n_prev_dirs = self.normalize_and_embed_previous_dirs(
                batch_t, unpack_results=True)
            n_prev_dirs = \
                self._stack_batch(n_prev_dirs, use_padding, batch_max_len)
            # Total input must still be of size d_model.
            inputs = torch.cat((inputs, n_prev_dirs), dim=-1)
        inputs = self.position_encoding_layer(inputs)

        # Targets
        targets = self._stack_batch(batch_t, use_padding, batch_max_len - 1)
        targets = self.embedding_layer_t(targets)
        targets = torch.cat((torch.full((len(batch_t), 1, self.d_model),
                                        fill_value=SOS_TOKEN),
                             targets), dim=1)
        targets = self.position_encoding_layer(targets)

        return inputs, targets

    def _run_main_layer_forward(
            self, embed_x: torch.Tensor, embed_t: torch.Tensor, masks: Tuple,
            return_weights: bool, average_heads: bool):
        raise NotImplementedError

    def compute_loss(self, model_outputs, streamlines, **kw):
        """
        Computes the loss function using the provided outputs and targets.
        Returns the mean loss (loss averaged across timesteps and sequences).

        Parameters
        ----------
        model_outputs : Any
            The model outputs for a batch of sequences. Ex: a gaussian mixture
            direction getter returns a Tuple[Tensor, Tensor, Tensor], but a
            cosine regression direction getter return a simple Tensor. Please
            make sure that the chosen direction_getter's output size fits with
            the target ou the target's data if it's a PackedSequence.
        streamlines : List
            The target values for the batch (the streamlines).

        Returns
        -------
        mean_loss : torch.Tensor
            The loss between the outputs and the targets, averaged across
            timesteps and sequences.
        """
        # Computing directions. Note that if previous dirs are used, this was
        # already computed when calling the forward method. We could try to
        # prevent double calculations, but a little complicated in actual class
        # structure.
        targets = compute_directions(streamlines)

        # Concatenating all points together to compute loss.
        return self.direction_getter.compute_loss(
            model_outputs.to(self.device), torch.cat(targets).to(self.device))

    def get_tracking_directions(self, model_outputs, algo):
        return self.direction_getter.get_tracking_directions(model_outputs,
                                                             algo)


class OriginalTransformerModel(AbstractTransformerModel):
    """
    We can use torch.nn.Transformer.
    We will also compare with
    https://github.com/jason9693/MusicTransformer-pytorch.git

                                                 direction getter
                                                        |
                                                     DECODER
                                                  --------------
                                                  |    Norm    |
                                                  |    Skip    |
                                                  |  Dropout   |
                                                  |2-layer FFNN|
                  ENCODER                         |      |     |
               --------------                     |    Norm    |
               |    Norm    | ---------           |    Skip    |
               |    Skip    |         |           |  Dropout   |
               |  Dropout   |         --------->  | Attention  |
               |2-layer FFNN|                     |      |     |
               |     |      |                     |   Norm     |
               |    Norm    |                     |   Skip     |
               |    Skip    |                     |  Dropout   |
               |  Dropout   |                     | Masked Att.|
               | Attention  |                     --------------
               --------------                            |
                     |                             emb_choice_y
                emb_choice_x

    """
    def __init__(self, experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedding_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 max_len: int, positional_encoding_key: str,
                 embedding_key_x: str, embedding_key_t: str,
                 # TRANSFORMER
                 d_model: int, ffnn_hidden_size: Union[int, None],
                 nheads: int, dropout_rate: float, activation: str,
                 norm_first: bool, n_layers_e: int, n_layers_d: int,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 normalize_targets: bool,
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Args
        ----
        n_layers_e: int
            Number of encoding layers in the encoder. [6]
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, nb_features, nb_previous_dirs,
                         prev_dirs_embedding_size, prev_dirs_embedding_key,
                         normalize_prev_dirs,
                         max_len, positional_encoding_key, embedding_key_x,
                         embedding_key_t, d_model, ffnn_hidden_size, nheads,
                         dropout_rate, activation, norm_first,
                         dg_key, dg_args, normalize_targets, neighborhood_type,
                         neighborhood_radius, log_level)

        # ----------- Additional params
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        logger.info("Instantiating torch transformer, may take a few "
                    "seconds...")
        # Encoder:
        encoder_layer = ModifiedTransformerEncoderLayer(
            self.d_model, self.nheads,
            dim_feedforward=self.ffnn_hidden_size, dropout=self.dropout_rate,
            activation=self.activation, batch_first=BATCH_FIRST,
            norm_first=self.norm_first)
        encoder = ModifiedTransformerEncoder(encoder_layer, n_layers_e,
                                             norm=None)

        # Decoder
        decoder_layer = ModifiedTransformerDecoderLayer(
            self.d_model, self.nheads,
            dim_feedforward=self.ffnn_hidden_size, dropout=self.dropout_rate,
            activation=self.activation, batch_first=BATCH_FIRST,
            norm_first=self.norm_first)
        decoder = ModifiedTransformerDecoder(decoder_layer, n_layers_d,
                                             norm=None)

        self.modified_torch_transformer = ModifiedTransformer(
            d_model, nheads, n_layers_e, n_layers_d, ffnn_hidden_size,
            dropout_rate, activation, encoder, decoder,
            batch_first=BATCH_FIRST,
            norm_first=self.norm_first)

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'n_layers_e': self.n_layers_e,
            'n_layers_d': self.n_layers_d,
        })
        return p

    def _run_main_layer_forward(self, embed_x, embed_t, masks,
                                return_weights, average_heads):
        """Original Main transformer

        Returns
        -------
        outputs: Tensor
            Shape: [nb_streamlines, max_batch_len, d_model]
        masks: Tuple
            Encoder's self-attention weights: [nb_streamlines, max_batch_len]
        """
        mask_future_x, mask_future_t, mask_padding = masks
        outputs, sa_weights_encoder, sa_weights_decoder, mha_weights = \
            self.modified_torch_transformer(
                src=embed_x, tgt=embed_t,
                src_mask=mask_future_x, tgt_mask=mask_future_t,
                memory_mask=mask_future_x,
                src_key_padding_mask=mask_padding,
                tgt_key_padding_mask=mask_padding,
                memory_key_padding_mask=mask_padding,
                return_weights=return_weights, average_heads=average_heads)
        return outputs, (sa_weights_encoder, sa_weights_decoder, mha_weights)


class TransformerSrcAndTgtModel(AbstractTransformerModel):
    """
    Decoder only. Concatenate source + target together as input.
    See https://arxiv.org/abs/1905.06596 and
    https://proceedings.neurips.cc/paper/2018/file/4fb8a7a22a82c80f2c26fe6c1e0dcbb3-Paper.pdf
    + discussion with Hugo.

                                                        direction getter
                                                              |
                                                  -------| take 1/2 |
                                                  |    Norm      x2 |
                                                  |    Skip      x2 |
                                                  |  Dropout     x2 |
                                                  |2-layer FFNN  x2 |
                                                  |        |        |
                                                  |   Norm       x2 |
                                                  |   Skip       x2 |
                                                  |  Dropout     x2 |
                                                  | Masked Att.  x2 |
                                                  -------------------
                                                           |
                                             [ emb_choice_x ; emb_choice_y ]

    """
    def __init__(self, experiment_name: str, nb_features: int,
                 # PREVIOUS DIRS
                 nb_previous_dirs: Union[int, None],
                 prev_dirs_embedding_size: Union[int, None],
                 prev_dirs_embedding_key: Union[str, None],
                 normalize_prev_dirs: bool,
                 # INPUTS
                 max_len: int, positional_encoding_key: str,
                 embedding_key_x: str, embedding_key_t: str,
                 # TRANSFORMER
                 d_model: int, ffnn_hidden_size: Union[int, None],
                 nheads: int, dropout_rate: float, activation: str,
                 norm_first: bool, n_layers_d: int,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: Union[dict, None],
                 normalize_targets: bool,
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Args
        ----
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(experiment_name, nb_features, nb_previous_dirs,
                         prev_dirs_embedding_size, prev_dirs_embedding_key,
                         normalize_prev_dirs,
                         max_len, positional_encoding_key, embedding_key_x,
                         embedding_key_t, d_model, ffnn_hidden_size, nheads,
                         dropout_rate, activation, norm_first,
                         dg_key, dg_args, normalize_targets, neighborhood_type,
                         neighborhood_radius, log_level)
        # ----------- Additional params
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        # We say "decoder only" from the logical point of view, but code-wise
        # it is actually "encoder only". A decoder would need output from the
        # encoder.
        logger.debug("Instantiating Transformer...")
        # The d_model is the same; points are not concatenated together.
        # It is the max_len that is modified: The sequences are concatenated
        # one beside the other.
        double_layer = ModifiedTransformerEncoderLayer(
            self.d_model, self.nheads,
            dim_feedforward=self.ffnn_hidden_size, dropout=self.dropout_rate,
            activation=self.activation, batch_first=True,
            norm_first=self.norm_first)
        self.modified_torch_transformer = ModifiedTransformerEncoder(
            double_layer, n_layers_d, norm=None)

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'n_layers_d': self.n_layers_d,
        })
        return p

    def _prepare_masks(self, unpadded_lengths, use_padding, batch_max_len):
        mask_future_x, mask_future_t, mask_padding = super()._prepare_masks(
            unpadded_lengths, use_padding, batch_max_len)

        # Concatenating mask future:
        #  - For the input: self-attention only.
        all_masked = torch.full((batch_max_len, batch_max_len), True)
        mask_future_x_padded = torch.cat((mask_future_x, all_masked), dim=-1)
        #  - For the target: mixed attention.
        mask_future_t_total = torch.cat((mask_future_x, mask_future_t), dim=-1)
        #  - Concat
        mask_future = torch.cat((mask_future_x_padded, mask_future_t_total),
                                dim=0)

        # Concatenating mask padding:
        if use_padding:
            mask_padding = torch.cat((mask_padding, mask_padding), dim=-1)

        return mask_future, mask_padding

    def _run_main_layer_forward(self, embed_x, embed_t, masks,
                                return_weights, average_heads):
        mask_future, total_mask_padding = masks

        # Concatenating x and t
        inputs = torch.cat((embed_x, embed_t), dim=1)

        # Main transformer
        outputs, sa_weights = self.modified_torch_transformer(
            src=inputs, mask=mask_future,
            src_key_padding_mask=total_mask_padding,
            return_weights=return_weights, average_heads=average_heads)

        # Take the second half of model outputs to direction getter
        # (the last skip-connection makes more sense this way. That's why it's
        # more a "decoder" than an "encoder" in logical meaning.
        kept_outputs = outputs[:, -outputs.shape[1]//2:, :]

        return kept_outputs, (sa_weights,)
