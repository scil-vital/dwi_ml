# -*- coding: utf-8 -*-
import logging
from time import time
from typing import Union, List, Tuple

from dipy.data import get_sphere
import numpy as np
import torch
from torch.nn import Dropout, Transformer
from torch.nn.functional import pad

from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, convert_dirs_to_class
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_directions
from dwi_ml.data.spheres import TorchSphere
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.main_models import (MainModelOneInput,
                                       ModelWithDirectionGetter,
                                       ModelWithNeighborhood)
from dwi_ml.models.projects.positional_encoding import \
    keys_to_positional_encodings
from dwi_ml.models.utils.transformers_from_torch import (
    ModifiedTransformer,
    ModifiedTransformerEncoder, ModifiedTransformerEncoderLayer,
    ModifiedTransformerDecoder, ModifiedTransformerDecoderLayer)

# Our model needs to be autoregressive, to allow inference / generation at
# tracking time.
# => During training, we hide the future; both in the input and in the target
# sequences.

# About the tracking process
# At each new step, the whole sequence is processed again (ran in the model).
# We only keep the last output. This is not very efficient... Is there a way
# to keep the hidden state in-between?
logger = logging.getLogger('model_logger')  # Same logger as Super.

# Trying to help with memory.
# When running out of memory, the error message is:
# torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate XX (GPU 0;
# X total capacity; X already allocated; X free; X reserved in total by Torch)
# If reserved memory is >> allocated memory try setting max_split_size_mb to
# avoid fragmentation. Value to which to limit is unclear.
# Tested, does not seem to improve much.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
CLEAR_CACHE = False


def forward_padding(data: torch.tensor, expected_length):
    return pad(data, (0, 0, 0, expected_length - len(data)))


def pad_and_stack_batch(data: List[torch.Tensor], pad_first: bool,
                        pad_length: int):
    """
    Pad the list of tensors so that all streamlines have length max_len.
    Then concatenate all streamlines.

    Params
    ------
    data: list[Tensor]
        Len: nb streamlines. Shape of each tensor: nb points x nb features.
    pad_first: bool
        If false, padding is skipped. (Ex: If all streamlines already
        contain the right number of points.)
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
        data = [forward_padding(data[i], pad_length) for i in range(len(data))]

    return torch.stack(data)


class AbstractTransformerModel(ModelWithNeighborhood, MainModelOneInput,
                               ModelWithDirectionGetter):
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
    def __init__(self,
                 experiment_name: str,
                 step_size: Union[float, None], compress: Union[float, None],
                 # INPUTS IN ENCODER
                 nb_features: int, embedding_key_x: str,
                 # TARGETS IN DECODER
                 token_type: str, embedding_key_t: str,
                 # GENERAL TRANSFORMER PARAMS
                 max_len: int, positional_encoding_key: str,
                 d_model: int, ffnn_hidden_size: Union[int, None],
                 nheads: int, dropout_rate: float, activation: str,
                 norm_first: bool,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: dict,
                 # Other
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, List[float], None],
                 log_level=logging.root.level):
        """
        Args
        ----
        nb_features: int
            This value should be known from the actual data. Number of features
            in the data (last dimension).
        embedding_key_x: str,
            Chosen class for the input embedding (the data embedding part).
            Choices: keys_to_embeddings.keys().
            Default: 'no_embedding'.
        token_type: str
            Either 'as_label' or the name of the sphere to convert to classes.
            Used for SOS addition.
        embedding_key_t: str,
            Target embedding, with the same choices as above.
            Default: 'no_embedding'.
        max_len: int
            Maximal sequence length. This is only used in the positional
            encoding. During the forward call, batches are only padded to the
            longest sequence in the batch. However, positional encoding only
            makes sence if not streamlines are longer than that value (this is
            verified).
        positional_encoding_key: str,
            Chosen class for the input's positional embedding. Choices:
            keys_to_positional_embeddings.keys(). Default: 'sinusoidal'.
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
        """
        super().__init__(
            # MainAbstract
            experiment_name=experiment_name, step_size=step_size,
            compress=compress, log_level=log_level,
            # Neighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            # Tracking
            dg_key=dg_key, dg_args=dg_args)

        self.nb_features = nb_features
        self.embedding_key_x = embedding_key_x
        self.token_type = token_type
        self.embedding_key_t = embedding_key_t
        self.max_len = max_len
        self.positional_encoding_key = positional_encoding_key
        self.d_model = d_model
        self.nheads = nheads
        self.ffnn_hidden_size = ffnn_hidden_size if ffnn_hidden_size \
            else d_model // 2
        self.dropout_rate = dropout_rate
        self.activation = activation
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
        assert d_model // nheads == float(d_model) / nheads, \
            "d_model ({}) must be divisible by nheads ({})"\
            .format(d_model, nheads)

        # ----------- Instantiations
        # This dropout is only used in the embedding; torch's transformer
        # prepares its own dropout elsewhere, and direction getter too.
        self.dropout = Dropout(self.dropout_rate)

        # 1. x embedding
        input_size = nb_features * (self.nb_neighbors + 1)
        cls_x = keys_to_embeddings[self.embedding_key_x]
        self.embedding_layer_x = cls_x(input_size, d_model)

        # 2. positional encoding
        cls_p = keys_to_positional_encodings[self.positional_encoding_key]
        self.position_encoding_layer = cls_p(d_model, dropout_rate, max_len)

        # 3. target embedding
        cls_t = keys_to_embeddings[self.embedding_key_t]
        if token_type == 'as_label':
            self.token_sphere = None
            target_features = 4
        else:
            dipy_sphere = get_sphere(token_type)
            self.token_sphere = TorchSphere(dipy_sphere)
            # nb classes = nb_vertices + SOS
            target_features = len(self.token_sphere.vertices) + 1
        self.embedding_layer_t = cls_t(target_features, d_model)

        # 4. Transformer: See child classes

        # 5. Direction getter
        # Original paper: last layer = Linear + Softmax on nb of classes.
        # Note on parameter initialization.
        # They all use torch.nn.linear, which initializes parameters based
        # on a kaiming uniform, same as uniform(-sqrt(k), sqrt(k)) where k is
        # the nb of features.
        self.instantiate_direction_getter(d_model)

        assert self.loss_uses_streamlines
        self.forward_uses_streamlines = True

    @property
    def params_for_checkpoint(self):
        """
        Every parameter necessary to build the different layers again
        from a checkpoint.
        """
        p = super().params_for_checkpoint
        p.update({
            'nb_features': int(self.nb_features),
            'embedding_key_x': self.embedding_key_x,
            'token_type': self.token_type,
            'embedding_key_t': self.embedding_key_t,
            'max_len': self.max_len,
            'positional_encoding_key': self.positional_encoding_key,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'nheads': self.nheads,
            'd_model': self.d_model,
            'ffnn_hidden_size': self.ffnn_hidden_size,
            'norm_first': self.norm_first,
        })

        return p

    def set_context(self, context):
        assert context in ['training', 'tracking', 'visu']
        self._context = context

    def move_to(self, device):
        super().move_to(device)
        if self.token_sphere is not None:
            self.token_sphere.move_to(device)

    def prepare_streamlines_f(self, streamlines):
        if self._context is None:
            raise ValueError("Please set the context before running the model."
                             "Ex: 'training'.")
        elif (self._context == 'training' or self._context == 'visu') and not \
                self.direction_getter.add_eos:
            # We don't use the last coord because does not have an associated
            # target direction (except if EOS is used).
            streamlines = [s[:-1, :] for s in streamlines]
        else:
            assert self._context in ['tracking', 'preparing_backward']
            # Reminder: during tracking, we keep all points.
            # For backward, we don't keep the last point (i.e. the seed)
            # because we will start from there.
            raise NotImplementedError("Streamline preparation for tracking "
                                      "is managed by the tracker!"
                                      "Code error!")

        return streamlines

    def _prepare_targets_forward(self, batch_streamlines):
        """
        batch_streamlines: List[Tensors]
        during_loss: bool
            If true, this is called during loss computation, and only EOS is
            added.
        during_foward: bool
            If True, this is called in the forward method, and both
            EOS and SOS are added.
        """
        batch_dirs = compute_directions(batch_streamlines)

        if self.token_type == 'as_label':
            batch_dirs = add_label_as_last_dim(batch_dirs,
                                               add_sos=True, add_eos=False)
        else:
            batch_dirs = convert_dirs_to_class(
                batch_dirs, self.token_sphere,
                add_sos=True, add_eos=False, to_one_hot=True)

        return batch_dirs

    def _generate_future_mask(self, sz):
        """DO NOT USE FLOAT, their code had a bug (see issue #92554. Fixed in
        latest GitHub branch. Waiting for release.) Using boolean masks.
        """
        mask = Transformer.generate_square_subsequent_mask(sz, self.device)
        return mask < 0

    def _generate_padding_mask(self, unpadded_lengths, batch_max_len):
        nb_streamlines = len(unpadded_lengths)

        mask_padding = torch.full((nb_streamlines, batch_max_len),
                                  fill_value=False, device=self.device)
        for i in range(nb_streamlines):
            mask_padding[i, unpadded_lengths[i]:] = True

        return mask_padding

    def _prepare_masks(self, unpadded_lengths, use_padding, batch_max_len):
        """
        Prepare masks for the transformer.

        Params
        ------
        unpadded_lengths: list
            Length of each streamline.
        use_padding: bool,
            If false, skip padding (all streamlines must have the same length).
        batch_max_len: int
            Batch's maximum length. It is not useful to pad more than that.
            (During tracking, particularly interesting!). Should be equal or
            smaller to self.max_len.

        Returns
        -------
        mask_future: Tensor
            Shape: [batch_max_len, batch_max_len]
            Masks the inputs that do not exist (useful for data generation, but
            also used during training on padded points because, why not), at
            each position.
            = [[False, True,  True],
               [False, False, True],
               [False, False, False]]
        mask_padding: Tensor
            Shape [nb_streamlines, batch_max_len]. Masks positions that do not
            exist in the sequence.
        """
        mask_future = self._generate_future_mask(batch_max_len)

        if use_padding:
            mask_padding = self._generate_padding_mask(unpadded_lengths,
                                                       batch_max_len)
        else:
            mask_padding = None

        return mask_future, mask_padding

    def forward(self, inputs: List[torch.tensor],
                batch_streamlines: List[torch.tensor], return_weights=False,
                average_heads=False):
        """
        Params
        ------
        inputs: list[Tensor]
            One tensor per streamline. Size of each tensor =
            [nb_input_points, nb_features].
        batch_streamlines: list[Tensor]
            Streamline coordinates. One tensor per streamline. Size of each
            tensor = [nb_input_points + 1, 3]. Directions will be computed to
            obtain targets of the same lengths. Then, directions are used for
            two things:
            - As input to the decoder. This input is generally the shifted
            sequence, with an SOS token (start of sequence) at the first
            position. In our case, there is no token, but the sequence is
            adequately masked to hide future positions. The last direction is
            not used.
            - As target during training. The whole sequence is used.
        return_weights: bool
            If true, returns the weights of the attention layers.
        average_heads: bool
            If return_weights, you may choose to average the weights from
            different heads together.

        Returns
        -------
        output: Tensor,
            Batch output, formatted differently based on context:
                - During training/visu:
                    [total nb points all streamlines, out size]
                - During tracking: [nb streamlines * 1, out size]
        weights: Tuple
            If return_weights: The weights (depending on the child model)
        """
        # Reminder.
        # Correct interpolation and management of points should be done
        # before. (Ex: by calling prepare_streamlines_f).

        if self._context is None:
            raise ValueError("Please set context before usage.")

        # Reminder. In all cases, len(each input) == len(each streamline).
        # Correct interpolation and management of points should be done
        # before. (Ex: by calling prepare_streamlines_f).
        assert np.all([len(i) == len(s) for i, s in
                       zip(inputs, batch_streamlines)])

        # Remember lengths to unpad outputs later.
        # (except during tracking, we only keep the last output, but still
        # verifying if any length exceeds the max allowed).
        unpadded_lengths = np.asarray([len(i) for i in inputs])

        # ----------- Checks
        if np.any(unpadded_lengths > self.max_len):
            raise ValueError("Some streamlines were longer than accepted max "
                             "length for sequences ({})".format(self.max_len))

        # ----------- Prepare masks and parameters
        # (Skip padding if all streamlines have the same length)
        use_padding = not np.all(unpadded_lengths == unpadded_lengths[0])
        batch_max_len = np.max(unpadded_lengths)
        if CLEAR_CACHE:
            now = time()
            logging.debug("Transformer: Maximal length in batch is {}"
                          .format(batch_max_len))
            torch.torch.cuda.empty_cache()
            now2 = time()
            logging.debug("Cleared cache in {} secs.".format(now2 - now))
        masks = self._prepare_masks(unpadded_lengths, use_padding,
                                    batch_max_len)

        # Compute targets (= directions).
        # Will be computed again later for loss computation, but ok, should not
        # be too heavy.
        targets = self._prepare_targets_forward(batch_streamlines)
        nb_streamlines = len(targets)

        # ----------- Ok. Start processing
        # Note. Tried calling torch.cuda.empty_cache() before.
        # Not really recommended, and does not seem to help much.
        # See many discussions in forums, such as
        # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/26

        # 1. Embedding + position encoding.
        # Run embedding on padded data. Necessary to make the model
        # adapt for the positional encoding.
        inputs, targets = self.run_embedding(inputs, targets, use_padding,
                                             batch_max_len)
        inputs, targets = self.dropout(inputs), self.dropout(targets)

        # 2. Main transformer
        outputs, weights = self._run_main_layer_forward(
            inputs, targets, masks, return_weights, average_heads)

        # Unpad now and combine everything for the direction getter.
        if self._context == 'tracking':
            outputs = outputs.detach()
            # No need to actually unpad, we only take the last (unpadded)
            # point, newly created. (-1 for python indexing)
            if use_padding:  # Not all the same length.
                outputs = [outputs[i, unpadded_lengths[i] - 1, :]
                           for i in range(nb_streamlines)]
                outputs = torch.vstack(outputs)
            else:  # All the same length (ex, during forward tracking)
                outputs = outputs[:, -1, :]
        else:
            # outputs size = [nb streamlines, max_len, d_model].
            if use_padding:
                # We take all (unpadded) points.
                outputs = [outputs[i, 0:unpadded_lengths[i], :]
                           for i in range(nb_streamlines)]
                outputs = torch.vstack(outputs)
            else:
                # We reshape the data directly. Last dimension = unchanged.
                # First dim: -1 = nb_streamlines * nb_points.
                outputs = torch.reshape(outputs, (-1, self.d_model))

        # 3. Direction getter
        # Outputs will be all streamlines merged.
        # To compute loss = ok. During tracking, we will need to split back.
        outputs = self.direction_getter(outputs)

        if return_weights:
            return outputs, weights

        return outputs

    def run_embedding(self, inputs, targets, use_padding, batch_max_len):
        """
        Pad + concatenate.
        Embedding. (Add SOS token to target.)
        Positional encoding.
        """
        # toDo: Test faster:
        #   1) stack (2D), embed, destack, pad_and_stack (3D)
        #   2) loop on streamline to embed, pad_and_stack
        #   3) pad_and_stack, then embed (but we might embed many zeros that
        #      will be masked in attention anyway)
        # Inputs
        inputs = pad_and_stack_batch(inputs, use_padding, batch_max_len)
        inputs = self.embedding_layer_x(inputs)
        inputs = self.position_encoding_layer(inputs)

        # Targets
        targets = pad_and_stack_batch(targets, use_padding, batch_max_len)
        targets = self.embedding_layer_t(targets)
        targets = self.position_encoding_layer(targets)

        return inputs, targets

    def _run_main_layer_forward(
            self, embed_x: torch.Tensor, embed_t: torch.Tensor, masks: Tuple,
            return_weights: bool, average_heads: bool):
        raise NotImplementedError

    def compute_loss(self, model_outputs, target_streamlines,
                     average_results=True, **kw):
        """
        Computes the loss function using the provided outputs and targets.

        Parameters
        ----------
        model_outputs : Any
            The model outputs for a batch of sequences. Ex: a gaussian mixture
            direction getter returns a Tuple[Tensor, Tensor, Tensor], but a
            cosine regression direction getter return a simple Tensor.
        target_streamlines : List
            The target values for the batch (the streamlines).
        average_results: bool
            If true, returns results averaged over timepoints (with the number
            of points). Else, returns all losses.
        """
        target_streamlines = self.direction_getter.prepare_targets_for_loss(
            target_streamlines)
        target_streamlines = torch.cat(target_streamlines, dim=0)

        # Concatenating all points together to compute loss.
        return self.direction_getter.compute_loss(
            model_outputs, target_streamlines, average_results)


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
    def __init__(self, n_layers_e: int, n_layers_d: int, **kw):
        """
        Args
        ----
        n_layers_e: int
            Number of encoding layers in the encoder. [6]
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(**kw)

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
            activation=self.activation, batch_first=True,
            norm_first=self.norm_first)
        encoder = ModifiedTransformerEncoder(encoder_layer, n_layers_e,
                                             norm=None)

        # Decoder
        decoder_layer = ModifiedTransformerDecoderLayer(
            self.d_model, self.nheads,
            dim_feedforward=self.ffnn_hidden_size, dropout=self.dropout_rate,
            activation=self.activation, batch_first=True,
            norm_first=self.norm_first)
        decoder = ModifiedTransformerDecoder(decoder_layer, n_layers_d,
                                             norm=None)

        self.modified_torch_transformer = ModifiedTransformer(
            self.d_model, self.nheads, n_layers_e, n_layers_d,
            self.ffnn_hidden_size, self.dropout_rate, self.activation,
            encoder, decoder, batch_first=True,
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
        # mask_future, mask_padding = masks

        outputs, sa_weights_encoder, sa_weights_decoder, mha_weights = \
            self.modified_torch_transformer(
                src=embed_x, tgt=embed_t,
                src_mask=masks[0], tgt_mask=masks[0], memory_mask=masks[0],
                src_key_padding_mask=masks[1], tgt_key_padding_mask=masks[1],
                memory_key_padding_mask=masks[1],
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
    def __init__(self, n_layers_d: int, **kw):
        """
        Args
        ----
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(**kw)

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
        mask_future, mask_padding = super()._prepare_masks(
            unpadded_lengths, use_padding, batch_max_len)

        # Concatenating mask future:
        #  - For the input: self-attention only. Do not look at target.
        all_masked = torch.full((batch_max_len, batch_max_len),
                                fill_value=True, device=self.device)
        mask_future_x = torch.cat((mask_future, all_masked), dim=-1)
        #  - For the target: mixed attention.
        mask_future_t = torch.cat((mask_future, mask_future), dim=-1)
        #  - Concat
        mask_future = torch.cat((mask_future_x, mask_future_t), dim=0)

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
        # more a "decoder" than an "encoder" in logical meaning.)
        kept_outputs = outputs[:, -outputs.shape[1]//2:, :]

        return kept_outputs, (sa_weights,)
