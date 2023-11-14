# -*- coding: utf-8 -*-
import logging
from time import time
from typing import Union, List, Tuple, Optional

from dipy.data import get_sphere
import numpy as np
import torch
from torch.nn import Dropout, Transformer
from torch.nn.functional import pad

from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, convert_dirs_to_class
from dwi_ml.data.processing.streamlines.post_processing import compute_directions
from dwi_ml.data.spheres import TorchSphere
from dwi_ml.models.embeddings import keys_to_embeddings
from dwi_ml.models.main_models import (ModelWithDirectionGetter,
                                       ModelWithNeighborhood,
                                       ModelOneInputWithEmbedding)
from dwi_ml.models.positional_encoding import keys_to_positional_encodings
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


class AbstractTransformerModel(ModelWithNeighborhood, ModelWithDirectionGetter,
                               ModelOneInputWithEmbedding):
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
                 # Target preprocessing params for the batch loader + tracker
                 step_size: Union[float, None],
                 compress_lines: Union[float, None],
                 # INPUTS IN ENCODER
                 nb_features: int, input_embedding_key: str,
                 input_embedded_size: int,
                 nb_cnn_filters: Optional[int], kernel_size: Optional[int],
                 # GENERAL TRANSFORMER PARAMS
                 max_len: int, positional_encoding_key: str,
                 ffnn_hidden_size: Union[int, None],
                 nheads: int, dropout_rate: float, activation: str,
                 norm_first: bool, n_layers_e,
                 # DIRECTION GETTER
                 dg_key: str, dg_args: dict,
                 # Other
                 neighborhood_type: Optional[str] = None,
                 neighborhood_radius: Optional[int] = None,
                 neighborhood_resolution: Optional[float] = None,
                 log_level=logging.root.level):
        """
        Note about embedding size:
            In the original model + SrcOnly model: defines d_model.
            (The transformer REQUIRES the same output dimension for each layer
            everywhere to allow skip connections. = d_model. Note that
            embeddings should also produce outputs of size d_model.)
            Default in original paper: 4096.

        Args
        ----
        max_len: int
            Maximal sequence length. This is only used in the positional
            encoding. During the forward call, batches are only padded to the
            longest sequence in the batch. However, positional encoding only
            makes sence if not streamlines are longer than that value (this is
            verified).
        positional_encoding_key: str,
            Chosen class for the input's positional embedding. Choices:
            keys_to_positional_embeddings.keys(). Default: 'sinusoidal'.
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
        n_layers_e: int
            All ours models have at least an encoder.
        """

        # Important. Super must be called first to verify input embedded size
        # through the ModelOneInputWithEmbedding.
        super().__init__(
            # MainAbstract
            experiment_name=experiment_name, step_size=step_size,
            compress_lines=compress_lines, log_level=log_level,
            # Neighborhood
            neighborhood_type=neighborhood_type,
            neighborhood_radius=neighborhood_radius,
            neighborhood_resolution=neighborhood_resolution,
            # For super ModelWithInputEmbedding:
            nb_features=nb_features,
            input_embedding_key=input_embedding_key,
            input_embedded_size=input_embedded_size,
            nb_cnn_filters=nb_cnn_filters, kernel_size=kernel_size,
            # Tracking
            dg_key=dg_key, dg_args=dg_args)

        self.max_len = max_len
        self.positional_encoding_key = positional_encoding_key
        self.nheads = nheads
        self.n_layers_e = n_layers_e  # All our models have an encoder
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.norm_first = norm_first
        self.ffnn_hidden_size = ffnn_hidden_size if ffnn_hidden_size is not None \
            else self.d_model // 2

        # ----------- Checks
        if self.d_model // self.nheads != float(self.d_model) / self.nheads:
            raise ValueError("d_model ({}) must be divisible by nheads ({})"
                             .format(self.d_model, self.nheads))

        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError('The dropout rate must be between 0 and 1.')

        if self.positional_encoding_key not in \
                keys_to_positional_encodings.keys():
            raise ValueError("Positional encoding choice not understood: {}"
                             .format(self.positional_encoding_key))

        # ----------- Instantiations
        # This dropout is only used in the embedding; torch's transformer
        # prepares its own dropout elsewhere, and direction getter too.
        self.dropout = Dropout(self.dropout_rate)

        # 1. x embedding layer
        assert self.computed_input_embedded_size > 3, \
            "Current computation of the positional encoding required data " \
            "of size > 3, but got {}".format(self.computed_input_embedded_size)

        # 2. positional encoding layer
        cls_p = keys_to_positional_encodings[self.positional_encoding_key]
        self.position_encoding_layer = cls_p(self.d_model, dropout_rate, max_len)

        # 3. target embedding layer: See child class with Target

        # 4. Transformer: See child classes

        # 5. Direction getter
        # Original paper: last layer = Linear + Softmax on nb of classes.
        # Note on parameter initialization.
        # They all use torch.nn.linear, which initializes parameters based
        # on a kaiming uniform, same as uniform(-sqrt(k), sqrt(k)) where k is
        # the nb of features.
        self.instantiate_direction_getter(self.d_model)

    @property
    def d_model(self):
        raise NotImplementedError

    @property
    def params_for_checkpoint(self):
        """
        Every parameter necessary to build the different layers again from a
        checkpoint.
        """
        p = super().params_for_checkpoint
        p.update({
            'nb_features': int(self.nb_features),
            'input_embedding_key': self.input_embedding_key,
            'input_embedded_size': self.input_embedded_size,
            'max_len': self.max_len,
            'n_layers_e': self.n_layers_e,
            'positional_encoding_key': self.positional_encoding_key,
            'dropout_rate': self.dropout_rate,
            'activation': self.activation,
            'nheads': self.nheads,
            'ffnn_hidden_size': self.ffnn_hidden_size,
            'norm_first': self.norm_first,
        })

        return p

    @classmethod
    def _load_params(cls, model_dir):
        params = super()._load_params(model_dir)

        # d_model now a property method.
        if 'd_model' in params:
            if isinstance(cls, TransformerSrcOnlyModel):
                params['input_embedded_size'] = params['d_model']

            del params['d_model']

        return params

    def set_context(self, context):
        # Training, validation: Used by trainer. Nothing special.
        # Tracking: Used by tracker. Returns only the last point.
        #     Preparing_backward: Used by tracker. Nothing special, but does
        #     not return only the last point.
        # Visu: Nothing special. Used by tester.
        # Visu_weights: Returns the weights too.
        assert context in ['training', 'validation', 'tracking',
                           'visu', 'visu_weights']
        self._context = context

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
                input_streamlines: List[torch.tensor] = None,
                average_heads=False):
        """
        Params
        ------
        inputs: list[Tensor]
            One tensor per streamline. Size of each tensor =
            [nb_input_points, nb_features].
        input_streamlines: list[Tensor]
            Streamline coordinates. One tensor per streamline. Size of each
            tensor = [nb_input_points, 3]. Directions will be computed to
            obtain targets of the same lengths. Then, directions are used for
            two things:
            - As input to the decoder. This input is generally the shifted
            sequence, with an SOS token (start of sequence) at the first
            position. In our case, there is no token, but the sequence is
            adequately masked to hide future positions. The last direction is
            not used.
            - As target during training. The whole sequence is used.
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
            If context is 'visu': The weights (depending on the child model)
        """
        if self.context is None:
            raise ValueError("Please set context before usage.")

        return_weights = False
        if self.context == 'visu_weights':
            return_weights = True

        # ----------- Checks
        if input_streamlines is not None:
            # If streamlines are necessary (depending on child class):
            # In all cases, len(each input) == len(each streamline).
            # Correct interpolation and management of points should be done
            # before.
            assert np.all([len(i) == len(s) for i, s in
                           zip(inputs, input_streamlines)])

        # Remember lengths to unpad outputs later.
        # (except during tracking, we only keep the last output, but still
        # verifying if any length exceeds the max allowed).
        input_lengths = np.asarray([len(i) for i in inputs])

        if np.any(input_lengths > self.max_len):
            raise ValueError("Some streamlines were longer than accepted max "
                             "length for sequences ({})".format(self.max_len))

        # ----------- Padding params
        use_padding = not np.all(input_lengths == input_lengths[0])
        batch_max_len = np.max(input_lengths)
        if CLEAR_CACHE:
            now = time()
            logging.debug("Transformer: Maximal length in batch is {}"
                          .format(batch_max_len))
            torch.torch.cuda.empty_cache()
            now2 = time()
            logging.debug("Cleared cache in {} secs.".format(now2 - now))

        # ----------- Prepare masks
        masks = self._prepare_masks(input_lengths, use_padding, batch_max_len)

        # Compute targets (= directions) for the decoder.
        nb_streamlines = len(inputs)

        # ----------- Ok. Start processing
        # Note. Tried calling torch.cuda.empty_cache() before.
        # Not really recommended, and does not seem to help much.
        # See many discussions in forums, such as
        # https://discuss.pytorch.org/t/about-torch-cuda-empty-cache/34232/26

        # 1. Embedding + position encoding.
        # Run embedding on padded data. Necessary to make the model
        # adapt for the positional encoding.
        data, constant_output = self._prepare_data(inputs, input_streamlines)
        data = self._run_embeddings(data, use_padding, batch_max_len)
        data = self._run_position_encoding(data)

        # 2. Main transformer
        outputs, weights = self._run_main_layer_forward(
            data, masks, return_weights, average_heads)

        # Here, data = one tensor, padded.
        # Unpad now and either
        #   a) combine everything for the direction getter, then unstack and
        #   restack when computing loss.  [Chosen here. See if we can improve]
        #   b) loop on direction getter. Stack when computing loss.
        # !!!     outputs size now  = [nb streamlines, max_len, d_model].
        #         input size to dg  = [nb points total, d_model]
        #         final output size = [nb points total, regression or
        #                                           classification output size]
        if self.context == 'tracking':
            # No need to actually unpad, we only take the last unpadded point
            # Ignoring both the beginning of the streamline (redundant from
            # previous tracking step) and the end of the streamline (padded
            # points).
            if use_padding:
                # Not all the same length (ex, backward tracking)
                # Taking output at the last coordinate = len(input) - 1
                # (-1 for python indexing)
                outputs = [outputs[i, input_lengths[i] - 1, :]
                           for i in range(nb_streamlines)]

                # Stacking for the direction getter.
                outputs = torch.vstack(outputs)

            else:
                # All the same length (ex, during forward tracking)
                # Keeping stacked.
                outputs = outputs[:, -1, :]

            # if start_from_copy_prev: using last one only
            # (could improve implementation here to skip it).
            # never padded
            if constant_output is not None:
                constant_output = [c[-1, :] for c in constant_output]
                constant_output = torch.vstack(constant_output)
        else:
            # We take all unpadded points.
            # Reminder. Each output will be the same length as the streamline,
            # i.e. one output per coordinate. It's the trainer's job to remove
            # the last coordinate if we don't need it (if no EOS).
            # Ignoring results at padded points.
            outputs = [outputs[i, 0:input_lengths[i], :]
                       for i in range(nb_streamlines)]

            # Stacking for the direction getter.
            outputs = torch.vstack(outputs)

            if constant_output is not None:  # ex, start_from_copy_prev:
                constant_output = torch.vstack(constant_output)

        # 3. Direction getter
        outputs = self.direction_getter(outputs)

        if constant_output is not None:
            outputs = constant_output + outputs

        # Splitting back. During tracking: only one point per streamline.
        if self.context != 'tracking':
            outputs = list(torch.split(outputs, list(input_lengths)))

        if return_weights:
            return outputs, weights

        return outputs

    def _prepare_data(self, inputs, input_streamlines):
        raise NotImplementedError

    def _run_embeddings(self, data, use_padding, batch_max_len):
        raise NotImplementedError

    def _run_position_encoding(self, data):
        raise NotImplementedError

    def _run_main_layer_forward(self, data, masks, return_weights,
                                average_heads):
        raise NotImplementedError

    def _run_input_embedding(self, inputs, use_padding, batch_max_len):
        # toDo: Test faster:
        #   1) stack (2D), embed, unstack, pad_and_stack (3D)
        #   2) loop on streamline to embed, pad_and_stack
        #   3) pad_and_stack, then embed (but we might embed many zeros that
        #      will be masked in attention anyway)

        # Inputs
        inputs = pad_and_stack_batch(inputs, use_padding, batch_max_len)
        inputs = self.input_embedding_layer(inputs)
        return inputs

    def merge_batches_outputs(self, all_outputs, new_batch, device=None):
        if self.context == 'visu_weights':
            new_outputs, new_weights = new_batch

            if all_outputs is None:
                outputs, weights = None, None
            else:
                outputs, weights = all_outputs
            new_outputs = super().merge_batches_outputs(outputs, new_outputs,
                                                        device)
            new_weights = self.merge_batches_weights(weights, new_weights,
                                                     device)
            return new_outputs, new_weights

        else:
            # No weights.
            return super().merge_batches_outputs(all_outputs, new_batch)

    def merge_batches_weights(self, weights, new_weights, device):
        raise NotImplementedError


class TransformerSrcOnlyModel(AbstractTransformerModel):
    def __init__(self, **kw):
        """
        No additional params. d_model = input_embedded_size.
        """
        super().__init__(**kw)

        # ----------- Additional instantiations
        logger.debug("Instantiating Transformer...")
        main_layer_encoder = ModifiedTransformerEncoderLayer(
            self.d_model, self.nheads, dim_feedforward=self.ffnn_hidden_size,
            dropout=self.dropout_rate, activation=self.activation,
            batch_first=True, norm_first=self.norm_first)
        self.modified_torch_transformer = ModifiedTransformerEncoder(
            main_layer_encoder, self.n_layers_e, norm=None)

    @property
    def d_model(self):
        # d_model is the same as the input size. Computed in
        # MainModelOneInputWithEmbedding
        return self.computed_input_embedded_size

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        return p

    @classmethod
    def _load_params(cls, model_dir):
        params = super()._load_params(model_dir)

        return params

    def _prepare_data(self, inputs, _):
        # Nothing to do. Ignoring targets.
        # No constant value to be added to output.
        return inputs, None

    def _run_embeddings(self, inputs, use_padding, batch_max_len):
        return self._run_input_embedding(inputs, use_padding, batch_max_len)
    
    def _run_position_encoding(self, inputs):
        inputs = self.position_encoding_layer(inputs)
        inputs = self.dropout(inputs)
        return inputs

    def _run_main_layer_forward(self, inputs, masks,
                                return_weights, average_heads):
        # Encoder only.

        # mask_future, mask_padding = masks
        outputs, sa_weights = self.modified_torch_transformer(
            src=inputs, mask=masks[0], src_key_padding_mask=masks[1],
            return_weights=return_weights, average_heads=average_heads)

        return outputs, (sa_weights,)

    def merge_batches_weights(self, weights, new_weights, device):
        # weights is a single attention tensor (encoder): a tuple of 1.
        new_weights = [a.to(device) for a in new_weights[0]]

        if weights is None:
            return (new_weights,)
        else:
            weights.extend(new_weights)
            return (weights,)


class AbstractTransformerModelWithTarget(AbstractTransformerModel):
    def __init__(self,
                 # TARGETS IN DECODER
                 sos_token_type: str, target_embedding_key: str,
                 target_embedded_size: int,
                 start_from_copy_prev: bool, **kwargs):
        """
        token_type: str
            Either 'as_label' or the name of the sphere to convert to classes.
            Used for SOS addition.
        target_embedding_key: str,
            Target embedding, with the same choices as above.
            Default: 'no_embedding'.
        """
        # Some checks before super init, in case d_model depends on target
        # embedded size.
        self.target_embedding_key = target_embedding_key
        self.target_embedded_size = target_embedded_size
        if sos_token_type == 'as_label':
            self.token_sphere = None
            self.target_features = 4
        else:
            dipy_sphere = get_sphere(sos_token_type)
            self.token_sphere = TorchSphere(dipy_sphere)
            # nb classes = nb_vertices + SOS
            self.target_features = len(self.token_sphere.vertices) + 1

        if self.target_embedding_key == 'no_embedding':
            if self.target_embedded_size is None:
                self.target_embedded_size = self.target_features
            assert self.target_embedded_size == self.target_features, \
                "With no_embedding for the target, input size must be equal " \
                "to the output embedded size. Expecting {}"\
                .format(self.target_features)
        else:
            assert self.target_embedding_key == 'nn_embedding', \
                "Unrecognized embedding key for the targets."

        super().__init__(**kwargs)

        self.sos_token_type = sos_token_type
        self.start_from_copy_prev = start_from_copy_prev

        # Checks.
        if self.target_embedding_key not in keys_to_embeddings.keys():
            raise ValueError("Embedding choice for targets not understood: {}"
                             .format(self.target_embedding_key))

        # 3. Target embedding.
        cls_t = keys_to_embeddings[self.target_embedding_key]
        self.embedding_layer_t = cls_t(self.target_features,
                                       self.target_embedded_size)

    @property
    def params_for_checkpoint(self):
        """
        Every parameter necessary to build the different layers again from a
        checkpoint.
        """
        p = super().params_for_checkpoint
        p.update({
            'sos_token_type': self.sos_token_type,
            'target_embedding_key': self.target_embedding_key,
            'target_embedded_size': self.target_embedded_size,
            'start_from_copy_prev': self.start_from_copy_prev
        })

        return p

    def move_to(self, device):
        super().move_to(device)
        if self.token_sphere is not None:
            self.token_sphere.move_to(device)

    def _prepare_data(self, inputs, input_streamlines):
        targets = compute_directions(input_streamlines)

        # Start from copy prev option.
        # Output is either itself or classes, but using output's sphere, not
        # necessarily the same as sos_token_sphere.
        copy_prev_dir = None
        if self.start_from_copy_prev:
            copy_prev_dir = self.format_prev_dir_(targets)
        if self.sos_token_type == 'as_label':
            targets = add_label_as_last_dim(targets,
                                            add_sos=True, add_eos=False)
        else:
            targets = convert_dirs_to_class(targets, self.token_sphere,
                                            add_sos=True, add_eos=False,
                                            to_one_hot=True)

        return (inputs, targets), copy_prev_dir

    def _run_embeddings(self, data, use_padding, batch_max_len):
        raise NotImplementedError

    def _run_position_encoding(self, data):
        raise NotImplementedError

    def _run_main_layer_forward(self, data, masks, return_weights,
                                average_heads):
        raise NotImplementedError

    def format_prev_dir_(self, dirs):
        """
        Format the previous direction at each point. (To add to output).
        At first coordinate: unkown. Using 0,0,0.

        If output is logits of classes: adding a one-hot vector with value
        6 on the right index (because sigmoid(6) is big). Always using value
        0 for the EOS class, if any.
        """
        if 'regression' in self.dg_key:
            # Regression: The latest previous dir will be used as skip
            # connection on the output.
            # Either take dirs and add [0, 0, 0] at each first position.
            # Or use pre-computed:
            copy_prev_dirs = dirs
        elif self.dg_key == 'sphere-classification':
            # Converting the input directions into classes the same way as
            # during loss, but convert to one-hot.
            # The first previous dir (0) converts to index 0.

            # Not necessarily the same class as previous dirs used as input to
            # the decoder.
            copy_prev_dirs = convert_dirs_to_class(
                dirs, self.direction_getter.torch_sphere, smooth_labels=False,
                add_sos=False, add_eos=False, to_one_hot=True)

            # Not adding a EOS point, but adding a EOS class with value 0.
            if self.direction_getter.add_eos:
                copy_prev_dirs = [torch.nn.functional.pad(cp, [0, 1, 0, 0])
                                  for cp in copy_prev_dirs]

            # Making the one from one-hot important for the sigmoid.
            copy_prev_dirs = [c * 6.0 for c in copy_prev_dirs]

        elif self.dg_key == 'smooth-sphere-classification':
            raise NotImplementedError
        elif 'gaussian' in self.dg_key:
            # The mean of the gaussian = the previous dir
            raise NotImplementedError
        else:
            # Fisher: not sure how to do that.
            raise NotImplementedError

        # Add zeros as previous dir at the first position
        copy_prev_dirs = [torch.nn.functional.pad(cp, [0, 0, 1, 0])
                          for cp in copy_prev_dirs]

        return copy_prev_dirs

    def _run_target_embedding(self, targets, use_padding, batch_max_len):
        targets = pad_and_stack_batch(targets, use_padding, batch_max_len)
        targets = self.embedding_layer_t(targets)

        return targets


class OriginalTransformerModel(AbstractTransformerModelWithTarget):
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
    def __init__(self, input_embedded_size, n_layers_d: int, **kw):
        """
        d_model = input_embedded_size = target_embedded_size.

        Args
        ----
        n_layers_d: int
            Number of encoding layers in the decoder. [6]
        """
        super().__init__(input_embedded_size=input_embedded_size,
                         target_embedded_size=input_embedded_size, **kw)

        # Veryfing that final computed values are still ok
        if self.computed_input_embedded_size != self.target_embedded_size:
            raise ValueError("For the original model, the input size and "
                             "target size after embedding must be equal "
                             "(value d_model) but got {} and {}"
                             .format(self.computed_input_embedded_size,
                                     self.target_embedded_size))

        # ----------- Additional params
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
        encoder = ModifiedTransformerEncoder(encoder_layer, self.n_layers_e,
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
            self.d_model, self.nheads, self.n_layers_e, n_layers_d,
            self.ffnn_hidden_size, self.dropout_rate, self.activation,
            encoder, decoder, batch_first=True,
            norm_first=self.norm_first)

    @property
    def d_model(self):
        # d_model = input size = target size
        return self.computed_input_embedded_size

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        del p['target_embedded_size']
        p['n_layers_d'] = self.n_layers_d
        return p

    def _run_embeddings(self, data, use_padding, batch_max_len):
        # input, targets = data
        inputs = self._run_input_embedding(data[0], use_padding, batch_max_len)
        targets = self._run_target_embedding(data[1], use_padding, batch_max_len)
        return inputs, targets

    def _run_position_encoding(self, data):
        # inputs, targets = data
        inputs = self.position_encoding_layer(data[0])
        inputs = self.dropout(inputs)

        targets = self.position_encoding_layer(data[1])
        targets = self.dropout(targets)

        return inputs, targets

    def _run_main_layer_forward(self, data, masks,
                                return_weights, average_heads):
        """Original Main transformer

        Returns
        -------
        outputs: Tensor
            Shape: [nb_streamlines, max_batch_len, d_model]
        masks: Tuple
            Encoder's self-attention weights: [nb_streamlines, max_batch_len]
        """
        # embed_x, embed_t = data
        # mask_future, mask_padding = masks
        outputs, sa_weights_encoder, sa_weights_decoder, mha_weights = \
            self.modified_torch_transformer(
                src=data[0], tgt=data[1],
                src_mask=masks[0], tgt_mask=masks[0], memory_mask=masks[0],
                src_key_padding_mask=masks[1], tgt_key_padding_mask=masks[1],
                memory_key_padding_mask=masks[1],
                return_weights=return_weights, average_heads=average_heads)
        return outputs, (sa_weights_encoder, sa_weights_decoder, mha_weights)

    def merge_batches_weights(self, weights, new_weights, device):
        # weights is a Tuple[encoder, decoder, cross]
        new_weights_e, new_weights_d, new_weights_c = new_weights
        new_weights_e = [a.to(device) for a in new_weights_e]
        new_weights_d = [a.to(device) for a in new_weights_d]
        new_weights_c = [a.to(device) for a in new_weights_c]

        if weights is None:
            return new_weights_e, new_weights_d, new_weights_c
        else:
            weights_e, weights_d, weights_c = weights
            weights_e.extend(new_weights_e)
            weights_d.extend(new_weights_d)
            weights_c.extend(new_weights_c)
            return weights_e, weights_d, weights_c


class TransformerSrcAndTgtModel(AbstractTransformerModelWithTarget):
    """
    Encoder only. Concatenate source + target together as input.
    See https://arxiv.org/abs/1905.06596 and
    https://proceedings.neurips.cc/paper/2018/file/4fb8a7a22a82c80f2c26fe6c1e0dcbb3-Paper.pdf
    + discussion with Hugo.

                                                  direction getter
                                                          |
                                                  |    Norm       |
                                                  |    Skip       |
                                                  |  Dropout      |
                                                  |2-layer FFNN   |
                                                  |       |       |
                                                  |   Norm        |
                                                  |   Skip        |
                                                  |  Dropout      |
                                                  | Masked Att.   |
                                                  -------------------
                                                           |
                                             [ emb_choice_x ; emb_choice_y ]

    """
    def __init__(self, **kw):
        """
        No additional params. d_model = input size + target size.
        """
        super().__init__(**kw)

        # ----------- Additional instantiations
        logger.debug("Instantiating Transformer...")
        main_layer_encoder = ModifiedTransformerEncoderLayer(
            self.d_model, self.nheads, dim_feedforward=self.ffnn_hidden_size,
            dropout=self.dropout_rate, activation=self.activation,
            batch_first=True, norm_first=self.norm_first)
        self.modified_torch_transformer = ModifiedTransformerEncoder(
            main_layer_encoder, self.n_layers_e, norm=None)

    @property
    def d_model(self):
        # d_model = input size = target size
        # target embedded size must be verified before super init.
        return self.computed_input_embedded_size + self.target_embedded_size

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        return p

    def _run_embeddings(self, data, use_padding, batch_max_len):
        # inputs, targets = data
        inputs = self._run_input_embedding(data[0], use_padding, batch_max_len)
        targets = self._run_target_embedding(data[1], use_padding, batch_max_len)
        inputs = torch.cat((inputs, targets), dim=-1)

        return inputs

    def _run_position_encoding(self, data):
        data = self.position_encoding_layer(data)
        data = self.dropout(data)
        return data

    def _run_main_layer_forward(self, concat_s_t, masks,
                                return_weights, average_heads):
        # Encoder only.

        # mask_future, mask_padding = masks
        outputs, sa_weights = self.modified_torch_transformer(
            src=concat_s_t, mask=masks[0], src_key_padding_mask=masks[1],
            return_weights=return_weights, average_heads=average_heads)

        return outputs, (sa_weights,)

    def merge_batches_weights(self, weights, new_weights, device):
        # weights is a single attention tensor (encoder)
        new_weights = [a.to(device) for a in new_weights]

        if weights is None:
            return new_weights
        else:
            weights.extend(new_weights)
            return weights
