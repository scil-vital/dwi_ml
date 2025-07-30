# -*- coding: utf-8 -*-
"""
Child classes of Torch Transformers. Changes are:

- Main transformer: Now returns the weights for visualisation + Added an option
  to decide if we want to share the linear weights for Q, K, V.
- Encoder: Idem
- Decoder: Idem

"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Transformer, TransformerDecoder, TransformerEncoder
from torch.nn.modules.transformer import _get_seq_len, _detect_is_causal_mask

from dwi_ml.models.projects.transformer_sublayers import \
    ModifiedTransformerDecoderLayer, ModifiedTransformerEncoderLayer

logger = logging.getLogger('model_logger')


class ModifiedTransformerEncoder(TransformerEncoder):
    def __init__(self, encoder_layer, *args, **kw):
        if not isinstance(encoder_layer, ModifiedTransformerEncoderLayer):
            raise ValueError("Encoder layer should be of type {}. Got {}"
                             .format(ModifiedTransformerEncoderLayer.__name__,
                                     type(encoder_layer)))
        super().__init__(encoder_layer, *args, **kw)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: Optional[bool] = None,
                # New args:
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(mask),
            other_name="mask",
            target_type=src.dtype
        )

        mask = F._canonical_mask(
            mask=mask,
            mask_name="mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        output = src
        convert_to_nested = False
        first_layer = self.layers[0]
        src_key_padding_mask_for_layers = src_key_padding_mask
        why_not_sparsity_fast_path = ''
        str_first_layer = "self.layers[0]"
        batch_first = first_layer.self_attn.batch_first
        if not hasattr(self, "use_nested_tensor"):
            why_not_sparsity_fast_path = "use_nested_tensor attribute not present"
        elif not self.use_nested_tensor:
            why_not_sparsity_fast_path = "self.use_nested_tensor (set in init) was not True"
        elif first_layer.training:
            why_not_sparsity_fast_path = f"{str_first_layer} was in training mode"
        elif not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif src_key_padding_mask is None:
            why_not_sparsity_fast_path = "src_key_padding_mask was None"
        elif (((not hasattr(self, "mask_check")) or self.mask_check)
                and not torch._nested_tensor_from_mask_left_aligned(src, src_key_padding_mask.logical_not())):
            why_not_sparsity_fast_path = "mask_check enabled, and src and src_key_padding_mask was not left aligned"
        elif output.is_nested:
            why_not_sparsity_fast_path = "NestedTensor input is not supported"
        elif mask is not None:
            why_not_sparsity_fast_path = "src_key_padding_mask and mask were both supplied"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"

        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                first_layer.self_attn.in_proj_weight,
                first_layer.self_attn.in_proj_bias,
                first_layer.self_attn.out_proj.weight,
                first_layer.self_attn.out_proj.bias,
                first_layer.norm1.weight,
                first_layer.norm1.bias,
                first_layer.norm2.weight,
                first_layer.norm2.bias,
                first_layer.linear1.weight,
                first_layer.linear1.bias,
                first_layer.linear2.weight,
                first_layer.linear2.bias,
            )
            _supported_device_type = ["cpu", "cuda", torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif src.device.type not in _supported_device_type:
                why_not_sparsity_fast_path = f"src device is neither one of {_supported_device_type}"
            elif torch.is_grad_enabled() and any(x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = ("grad is enabled and at least one of query or the "
                                              "input/output projection weights or biases requires_grad")

            if (not why_not_sparsity_fast_path) and (src_key_padding_mask is not None):
                convert_to_nested = True
                output = torch._nested_tensor_from_mask(output, src_key_padding_mask.logical_not(), mask_check=False)
                src_key_padding_mask_for_layers = None

        seq_len = _get_seq_len(src, batch_first)
        is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)

        # THIS IS THE MODIFIED PART
        sa_weights = [None] * len(self.layers)
        for mod, i in zip(self.layers, range(len(self.layers))):
            output, sa_weights[i] = mod(
                output, src_mask=mask, is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
        # END OF MODIFIED PART

        if convert_to_nested:
            output = output.to_padded_tensor(0., src.size())

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_weights


class ModifiedTransformerDecoder(TransformerDecoder):

    def __init__(self, decoder_layer, *args, **kw):
        if not isinstance(decoder_layer, ModifiedTransformerDecoderLayer):
            raise ValueError("Encoder layer should be of type {}. Got {}"
                             .format(ModifiedTransformerEncoderLayer.__name__,
                                     type(decoder_layer)))
        super().__init__(decoder_layer, *args, **kw)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_is_causal: Optional[bool] = None,
                memory_is_causal: bool = False,
                # New args:
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        output = tgt

        seq_len = _get_seq_len(tgt, self.layers[0].self_attn.batch_first)
        tgt_is_causal = _detect_is_causal_mask(tgt_mask, tgt_is_causal, seq_len)

        # THIS IS THE MODIFIED PART
        mha_weights = [None] * len(self.layers)
        sa_weights = [None] * len(self.layers)
        for mod, i in zip(self.layers, range(len(self.layers))):
            output, mha_weights[i], sa_weights[i] = \
                mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    tgt_is_causal=tgt_is_causal,
                    memory_is_causal=memory_is_causal,
                    # New args:
                    return_weights=return_weights,
                    average_heads=average_heads)
        # END OF MODIFIED PART

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_weights, mha_weights


class ModifiedTransformer(Transformer):
    encoder: ModifiedTransformerEncoder
    decoder: ModifiedTransformerDecoder

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None, memory_mask: Tensor = None,
                src_key_padding_mask: Tensor = None,
                tgt_key_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None,
                src_is_causal: bool = None, tgt_is_causal: bool = None,
                memory_is_causal: bool = False,
                # New args:
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        logger.debug("Entering main Transformer's forward.")
        memory, sa_weights_encoder = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            is_causal=src_is_causal,
            # New args:
            return_weights=return_weights, average_heads=average_heads)

        output, sa_weights_decoder, mha_weights = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            tgt_is_causal=tgt_is_causal, memory_is_causal=memory_is_causal,
            # New args:
            return_weights=return_weights, average_heads=average_heads)

        return output, sa_weights_encoder, sa_weights_decoder, mha_weights
