# -*- coding: utf-8 -*-
import logging
from typing import Union, List

import numpy as np
import torch
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from torch.nn import Dropout
from torch.nn.functional import pad
from torch.nn.modules.transformer import (
    Transformer,
    TransformerEncoder, TransformerDecoder,
    TransformerEncoderLayer, TransformerDecoderLayer)
from torch.nn.utils.rnn import pack_sequence, PackedSequence, unpack_sequence

from dwi_ml.models.main_models import (MainModelOneInput,
                                       ModelForTracking,
                                       ModelWithPreviousDirections,
                                       ModelWithNeighborhood)
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.direction_getter_models import keys_to_direction_getters
from dwi_ml.models.projects.positional_encoding import \
    keys_to_positional_encodings

# Pour les masques:
# https://stackoverflow.com/questions/68205894/how-to-prepare-data-for-tpytorchs-3d-attn-mask-argument-in-multiheadattention
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html

logger = logging.getLogger('model_logger')  # Same logger as Super.


def forward_padding(data: torch.tensor, nb_pad):
    # data: tensor of size
    return pad(data, (0, 0, 0, nb_pad))


class AbstractTransformerModel(ModelWithPreviousDirections,
                               ModelWithNeighborhood,
                               MainModelOneInput, ModelForTracking):
    """
    Prepares the parts common to our two transformer versions: embeddings,
    direction getter and some parameters for the model.

    Encoder and decoder will be prepared in child classes.

    Child forward methods should look like:
        x, t = self._run_embeddings(x, t)
        outputs = (run main transformer)
        formatted_outputs = self.direction_getter(outputs)

    About data embedding:
    We could even use the raw data, technically. But when adding the positional
    embedding, the reason it works is that the learning of the embedding
    happens while knowing that some positional vector will be added to it.
    As stated in the blog
    https://kazemnejad.com/blog/transformer_architecture_positional_encoding/
    the embedding probably adapts to leave place for the positional embedding.

    All this to say we can't use raw data, and the minimum is to learn to adapt
    with a neural network.
    """
    layer_norm = 1e-5  # epsilon value for the normalization sub-layers
    norm_first = False  # If True, encoder and decoder layers will perform
    # LayerNorms before other attention and feedforward operations, otherwise
    # after. Torch default + in original paper: False.
    batch_first = True  # If True, then the input and output tensors are
    # provided as (batch, seq, feature). If False, (seq, batch, feature).

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
            add_vectors_to_data=False,
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
            "d_model ({}) must be divisible by nheads ({})" \
            .format(d_model, nheads)

        # ----------- Input size:
        # (neighborhood prepared by super)
        nb_neighbors = len(self.neighborhood_vectors) if \
            self.neighborhood_vectors else 0
        self.input_size = nb_features * (nb_neighbors + 1)

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
        })
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
        })
        return p

    def _pad_and_stack_batch(self, unpadded_data, padding_necessary=True):
        """
        Pad the batch inputs and targets so that all streamlines have lenth
        max_len. Then concatenate all streamlines.

        Params
        ------
        batch_x: list[Tensor]
            Len: nb streamlines. Shape of each tensor: nb points x nb features.

        Returns
        -------
        formatted_x: Tensor
            Shape [nb_streamlines, max_len, nb_features] where nb features is
            the size of the batch input at this point (ex, initial number of
            features or d_model if embedding is already done).
        """
        if padding_necessary:
            padded_data = []
            for i in range(len(unpadded_data)):
                nb_pad_to_add = self.max_len - len(unpadded_data[i])
                padded_x = forward_padding(unpadded_data[i], nb_pad_to_add)
                # shape -> [max_len, nb_features]
                padded_data.append(padded_x)
        else:
            padded_data = unpadded_data

        return torch.stack(padded_data).to(self.device)

    def _prepare_masks(self, unpadded_lengths):
        """
        Prepare masks for the transformer.

        Params
        ------
        batch_x: list
            List of the upadded inputs lengths.

        Returns
        -------
        mask_future: Tensor
            Shape: [max_len, max_len]
            Masks the inputs that do not exist (useful for data generation, but
            also used during training), at each position.
            = [[0, -inf, -inf],
               [0,  0,  -inf],
               [0,  0,   0]]
        mask_padding: Tensor
            Shape [nb_streamlines, max_len]
            Masks the zeros in the padded sequences.

        """
        nb_streamlines = len(unpadded_lengths)

        mask_future = Transformer.generate_square_subsequent_mask(self.max_len)
        mask_padding = torch.zeros(nb_streamlines, self.max_len)
        for i in range(nb_streamlines):
            mask_padding[i, unpadded_lengths[i]:] = float('-inf')

        return mask_future, mask_padding

    def forward(self, batch_x, batch_streamlines, version=1):
        """
        Params
        ------
        batch_x: list[Tensor]
            One tensor per streamline. Size of each tensor =
            [nb_input_points, nb_features].
        batch_streamlines: list[Tensor]
            Streamline coordinates. One tensor per streamline. Size of each
            tensor = [nb_input_points + 1, 3].
        version: toDo, to be tested. VERSION 2 NOT WORKING; the backward fails.
        """
        # Compute targets (= directions).
        # Will be computed again later for loss computation, but ok, should not
        # be too heavy.
        batch_t = compute_directions(batch_streamlines, self.device)

        # Remember lengths to unpad later
        unpadded_lengths = np.asarray([len(i) for i in batch_t])

        # Padding unnecessary if all streamlines were resampled to max_len.
        padding_necessary = True if np.any(unpadded_lengths - self.max_len) \
            else False
        if not padding_necessary:
            logger.debug("Good news. All streamlines already had expected "
                         "length. No padding added.")

        # Embedding + position encoding.
        logger.debug("*** 1. Embedding and position encoding.")
        import time
        start = time.time()
        if version == 1:
            # We could do like in Learn2track and run the embedding on the
            # PackedSequence's data. It would run faster (no need to run a
            # bunch of zeros) + easier to concatenate with prev dirs.
            embed_x, embed_t = self.run_embedding_version1(batch_x, batch_t,
                                                           padding_necessary)
        else:
            # Or, we do like in original paper. Run on padded data. As
            # explained in doc above, probably necessary to make the model
            # learn embedding on whole padded sequence so that it can adapt for
            # the positional encoding.
            embed_x, embed_t = self.run_embedding_version2(batch_x, batch_t,
                                                           padding_necessary)
        end = time.time()
        print(end - start)
        # version 1: ~0.0015 with 16 short streamlines
        # version 2: ~0.0006 with 16 short streamlines.
        # todo. Test with bigger batches.
        embed_x = self.dropout(embed_x)
        embed_t = self.dropout(embed_t)

        logger.debug("Embedded inputs padded size: {}".format(embed_x.shape))
        logger.debug("Embedded targets padded size: {}".format(embed_t.shape))

        # Prepare masks
        mask_future, mask_padding = self._prepare_masks(unpadded_lengths)

        # Now run main transformer (see child classes)
        logger.debug("*** 2. Transformer....")
        outputs = self._run_main_layer_forward(embed_x, embed_t, mask_future,
                                               mask_padding)

        # Version 1: Unpad now and loop on streamlines for the direction getter
        # Version 2: Call direction getter once (but run on zeros) and unpad
        # after (during training, the model will see a lot of zeros.)
        # Tested on 16 short streamlines, version 1 seems faster + more
        # logical. Keeping.

        if padding_necessary:
            unpadded_outputs = [outputs[i][0:unpadded_lengths[i]]
                                for i in range(len(outputs))]
        else:
            unpadded_outputs = outputs

        # Direction getter
        logger.debug("*** 3. Direction getter (on unpadded data)....")
        final_outputs = []
        for i in range(len(outputs)):
            final_outputs.append(
                self.direction_getter(unpadded_outputs[i]))

        return final_outputs

    def run_embedding_version1(self, batch_x, batch_t, padding_necessary=True):
        """
        Version one.

        1) Packing
        2) Run embeddings on packed sequences (allows to run one shot)
        3) Concatenate prev_dirs + input (one shot)
        4) Padding + concatenating all streamlines.
        5) Positional embedding.
        """
        # Packing inputs and saving info
        inputs = pack_sequence(batch_x, enforce_sorted=False).to(self.device)
        targets = pack_sequence(batch_t, enforce_sorted=False).to(self.device)
        batch_sizes = inputs.batch_sizes
        sorted_indices = inputs.sorted_indices
        unsorted_indices = inputs.unsorted_indices
        # nb_input_points = len(inputs.data)

        logger.debug("    1.A. Embeddings (on packed_sequence's tensor!)...")
        inputs = self.embedding_layer_x(inputs.data)
        targets = self.embedding_layer_t(targets.data)
        n_prev_dirs = self.normalize_and_embed_previous_dirs(
            batch_t, unpack_results=False)

        logger.debug("    1.B. Concatenating previous dirs and inputs's "
                     "embeddings...")
        if n_prev_dirs is not None:
            inputs = torch.cat((inputs, n_prev_dirs), dim=-1)

        # Unpacking
        inputs = PackedSequence(inputs, batch_sizes, sorted_indices,
                                unsorted_indices)
        inputs = unpack_sequence(inputs)
        targets = PackedSequence(targets, batch_sizes, sorted_indices,
                                 unsorted_indices)
        targets = unpack_sequence(targets)

        logger.debug("    1.C. Padding and concatenating streamlines...")
        inputs = self._pad_and_stack_batch(inputs, padding_necessary)
        targets = self._pad_and_stack_batch(targets, padding_necessary)

        logger.debug("    1.D. Positional encoding")
        inputs = self.position_encoding_layer(inputs)
        targets = self.position_encoding_layer(targets)

        return inputs, targets

    def run_embedding_version2(self, batch_x, batch_t, padding_necessary=True):
        """
        Version 2:
        1. Pad + embedding.
        3. Combine with prev dirs
        4. Positional encoding.
        """

        logger.debug("    1.A. Padding and concatenating streamlines.")
        inputs = self._pad_and_stack_batch(batch_x, padding_necessary)
        targets = self._pad_and_stack_batch(batch_t, padding_necessary)

        logger.debug("    1.B. Run embedding (on padded sequences)...")
        inputs = self.embedding_layer_x(inputs)
        targets = self.embedding_layer_t(targets)
        n_prev_dirs = self.normalize_and_embed_previous_dirs(
            batch_t, unpack_results=True)  # Compute from raw dirs.

        logger.debug("    1.C. Combine with inputs...")
        if n_prev_dirs is not None:
            n_prev_dirs = self._pad_and_stack_batch(n_prev_dirs,
                                                    padding_necessary)
            inputs = torch.cat((inputs, n_prev_dirs), dim=-1)

        logger.debug("    1.D. positional encoding.")
        inputs = self.position_encoding_layer(inputs)
        targets = self.position_encoding_layer(targets)

        return inputs, targets

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_masks):
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

        # Packing values and using the .data, or looping on streamlines?
        # On testing data: packing = 0.012, looping = 0.002
        # toDo verify this on larger data? Also, it changes the computation
        #  of the backward, so maybe verify the whole execution's time?
        nb_streamlines = len(streamlines)
        mean_loss = torch.tensor(0.)
        for i in range(nb_streamlines):
            # Computing loss
            mean_loss += self.direction_getter.compute_loss(
                model_outputs[i].to(self.device), targets[i].to(self.device))

        mean_loss /= nb_streamlines

        return mean_loss

    def get_tracking_direction_det(self, model_outputs):
        self.direction_getter.get_tracking_direction_det(model_outputs)

    def sample_tracking_direction_prob(self, model_outputs):
        self.direction_getter.sample_tracking_direction_prob(
            model_outputs)


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
                 n_layers_e: int, n_layers_d: int,
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
                         dropout_rate, activation, dg_key, dg_args,
                         normalize_targets, neighborhood_type,
                         neighborhood_radius, log_level)

        # ----------- Additional params
        self.n_layers_e = n_layers_e
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        logger.info("Instantiating torch transformer, may take a few "
                    "seconds...")
        # Encoder:
        encoder_layer = TransformerEncoderLayer(
            self.d_model, self.nheads, self.ffnn_hidden_size,
            self.dropout_rate, self.activation, batch_first=True,
            norm_first=self.norm_first)
        encoder = TransformerEncoder(encoder_layer, n_layers_e, norm=None)

        # Decoder
        decoder_layer = TransformerDecoderLayer(
            self.d_model, self.nheads, self.ffnn_hidden_size,
            self.dropout_rate, self.activation, batch_first=True,
            norm_first=self.norm_first)
        decoder = TransformerDecoder(decoder_layer, n_layers_d, norm=None)

        self.transformer_layer = Transformer(
            d_model, nheads, n_layers_e, n_layers_d, ffnn_hidden_size,
            dropout_rate, activation, encoder, decoder,
            self.layer_norm, batch_first=True, norm_first=False)

    @property
    def params_for_json_prints(self):
        p = super().params_for_json_prints
        p.update({
            'n_layers_e': self.n_layers_e,
            'n_layers_d': self.n_layers_d,
        })
        return p

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'n_layers_e': self.n_layers_e,
            'n_layers_d': self.n_layers_d,
        })
        return p

    def _run_main_layer_forward(self, embed_x, embed_t, mask, padded_mask):
        """Original Main transformer"""

        outputs = self.transformer_layer(
            src=embed_x, tgt=embed_t,
            src_mask=mask, tgt_mask=mask, memory_mask=mask,
            src_key_padding_mask=padded_mask,
            tgt_key_padding_mask=padded_mask,
            memory_key_padding_mask=padded_mask)

        return outputs


class TransformerSrcAndTgtModel(AbstractTransformerModel):
    """
    Decoder only. Concatenate source + target together as input.
    See https://arxiv.org/abs/1905.06596
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
                 n_layers_d: int,
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
                         dropout_rate, activation, dg_key, dg_args,
                         normalize_targets, neighborhood_type,
                         neighborhood_radius, log_level)

        # ----------- Additional params
        self.n_layers_d = n_layers_d

        # ----------- Additional instantiations
        # We say "decoder only" from the logical point of view, but code-wise
        # it is actually "encoder only". A decoder would need output from the
        # encoder.
        logger.debug("Instantiating Transformer...")
        double_layer = TransformerEncoderLayer(
            self.d_model * 2, self.nheads, self.ffnn_hidden_size,
            self.dropout_rate, self.activation, batch_first=True,
            norm_first=self.norm_first)
        self.main_layer = TransformerEncoder(double_layer, n_layers_d,
                                             norm=None)

    @property
    def params_for_json_prints(self):
        p = super().params_for_json_prints
        p.update({
            'n_layers_d': self.n_layers_d,
        })
        return p

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'n_layers_d': self.n_layers_d,
        })
        return p

    def _run_main_layer_forward(self, embed_x, embed_t, future_mask,
                                padded_mask):
        # Concatenating x and t
        inputs = torch.cat((embed_x, embed_t), dim=-1)
        logger.debug("Concatenated [src | tgt] shape: {}".format(inputs.shape))

        # Main transformer
        outputs = self.main_layer(src=inputs, mask=future_mask,
                                  src_key_padding_mask=padded_mask)
        logger.debug("Outputs shape: {}".format(outputs.shape))

        # Take the second half of model outputs to direction getter
        # (the last skip-connection makes more sense this way. That's why it's
        # more a "decoder" than an "encoder" in logical meaning.
        kept_outputs = outputs[:, :, -self.d_model:]
        logger.debug("Final outputs shape: {}".format(kept_outputs.shape))
        return kept_outputs
