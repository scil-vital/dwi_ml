"""
Child classes of Torch Transformers. Changes are:

- EncoderLayer: Idem
- DecoderLayer: Idem

"""
import logging
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import (TransformerDecoderLayer, TransformerEncoderLayer,
                      MultiheadAttention, Parameter)

logger = logging.getLogger('model_logger')


def do_not_share_linear_weights(attn: MultiheadAttention, d_model):
    """
    I added a request for this parameter to be accessible.
    https://github.com/pytorch/pytorch/issues/92990

    Copied from MultiheadAttention's init method
    """

    factory_kwargs = {'device': None, 'dtype': None}

    # Overriding some parameters in the self attention.
    # Ugly but.... Torch does not have a parameter to NOT share linear
    # weights. In their code, their only NOT share weights when dimensions
    # are not the same. This is not our case. This is saved in their
    # parameter _qkv_same_embed_dim. By changing this, we change their
    # forward call to the MultiHeadAttention in self.self_attn.
    attn._qkv_same_embed_dim = False
    attn.q_proj_weight = Parameter(
        torch.empty((d_model, d_model), **factory_kwargs))
    attn.k_proj_weight = Parameter(
        torch.empty((d_model, d_model), **factory_kwargs))
    attn.v_proj_weight = Parameter(
        torch.empty((d_model, d_model), **factory_kwargs))
    attn.register_parameter('in_proj_weight', None)
    attn._reset_parameters()


class ModifiedTransformerEncoderLayer(TransformerEncoderLayer):
    def __init__(self, d_model, nhead, **kw):
        super().__init__(d_model, nhead, **kw)

        do_not_share_linear_weights(self.self_attn, d_model)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                is_causal: bool = False,
                # New args:
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        why_not_sparsity_fast_path = ''
        if not src.dim() == 3:
            why_not_sparsity_fast_path = f"input not batched; expected src.dim() of 3 but got {src.dim()}"
        elif self.training:
            why_not_sparsity_fast_path = "training is enabled"
        elif not self.self_attn.batch_first:
            why_not_sparsity_fast_path = "self_attn.batch_first was not True"
        elif not self.self_attn._qkv_same_embed_dim:
            why_not_sparsity_fast_path = "self_attn._qkv_same_embed_dim was not True"
        elif not self.activation_relu_or_gelu:
            why_not_sparsity_fast_path = "activation_relu_or_gelu was not True"
        elif not (self.norm1.eps == self.norm2.eps):
            why_not_sparsity_fast_path = "norm1.eps is not equal to norm2.eps"
        elif src.is_nested and (
                src_key_padding_mask is not None or src_mask is not None):
            why_not_sparsity_fast_path = "neither src_key_padding_mask nor src_mask are not supported with NestedTensor input"
        elif self.self_attn.num_heads % 2 == 1:
            why_not_sparsity_fast_path = "num_head is odd"
        elif torch.is_autocast_enabled():
            why_not_sparsity_fast_path = "autocast is enabled"
        if not why_not_sparsity_fast_path:
            tensor_args = (
                src,
                self.self_attn.in_proj_weight,
                self.self_attn.in_proj_bias,
                self.self_attn.out_proj.weight,
                self.self_attn.out_proj.bias,
                self.norm1.weight,
                self.norm1.bias,
                self.norm2.weight,
                self.norm2.bias,
                self.linear1.weight,
                self.linear1.bias,
                self.linear2.weight,
                self.linear2.bias,
            )

            # We have to use list comprehensions below because TorchScript does not support
            # generator expressions.
            _supported_device_type = ["cpu", "cuda",
                                      torch.utils.backend_registration._privateuse1_backend_name]
            if torch.overrides.has_torch_function(tensor_args):
                why_not_sparsity_fast_path = "some Tensor argument has_torch_function"
            elif not all((x.device.type in _supported_device_type) for x in
                         tensor_args):
                why_not_sparsity_fast_path = (
                    "some Tensor argument's device is neither one of "
                    f"{_supported_device_type}")
            elif torch.is_grad_enabled() and any(
                    x.requires_grad for x in tensor_args):
                why_not_sparsity_fast_path = (
                    "grad is enabled and at least one of query or the "
                    "input/output projection weights or biases requires_grad")

            if not why_not_sparsity_fast_path:
                merged_mask, mask_type = self.self_attn.merge_masks(src_mask,
                                                                    src_key_padding_mask,
                                                                    src)
                # MODIFIED:
                if return_weights:
                    raise NotImplementedError(
                        "Did not expect to reach here. Not ready to return "
                        "weights. Please contact dwi_ml developpers")
                return torch._transformer_encoder_layer_fwd(
                    src,
                    self.self_attn.embed_dim,
                    self.self_attn.num_heads,
                    self.self_attn.in_proj_weight,
                    self.self_attn.in_proj_bias,
                    self.self_attn.out_proj.weight,
                    self.self_attn.out_proj.bias,
                    self.activation_relu_or_gelu == 2,
                    self.norm_first,
                    self.norm1.eps,
                    self.norm1.weight,
                    self.norm1.bias,
                    self.norm2.weight,
                    self.norm2.bias,
                    self.linear1.weight,
                    self.linear1.bias,
                    self.linear2.weight,
                    self.linear2.bias,
                    merged_mask,
                    mask_type,
                )

        x = src
        if self.norm_first:
            # Norm, SA, Add, Norm, FF, Add
            sa, sa_weights = self._sa_block(
                self.norm1(x), src_mask, src_key_padding_mask,
                is_causal=is_causal,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            # SA, Add, Norm, FF, Add, Norm
            sa, sa_weights = self._sa_block(
                x, src_mask, src_key_padding_mask, is_causal=is_causal,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x, sa_weights

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False,
                  # New args:
                  return_weights=False, average_heads=False):
        x, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            # Modified args:
            need_weights=return_weights, average_attn_weights=average_heads)

        return self.dropout1(x), weights


class ModifiedTransformerDecoderLayer(TransformerDecoderLayer):
    """
    Decoder Layer, in the case where we do not have a start of sequence (SOS)
    token, and our mask contains only -inf for the first position. Output of
    self-attention becomes nan after the softmax step. Setting to 0.

    Also, now returning attention weights.
    """
    def __init__(self, d_model, nhead, **kw):
        super().__init__(d_model, nhead, **kw)

        do_not_share_linear_weights(self.self_attn, d_model)
        do_not_share_linear_weights(self.multihead_attn, d_model)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Tensor = None, memory_mask: Tensor = None,
                tgt_key_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None,
                tgt_is_causal: bool = False,
                memory_is_causal: bool = False,
                # New args:
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights + converts nan to 0.
        Weights are None if return_weights is False.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        if self.norm_first:
            # Norm, SA, Add, Norm, MHA, Add, Norm, FF, Add
            sa, sa_weights = self._sa_block(
                self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
            x = x + sa

            mha, mha_weights = self._mha_block(
                self.norm2(x), memory, memory_mask, memory_key_padding_mask,
                memory_is_causal,
                # Nre args:
                return_weights=return_weights, average_heads=average_heads)
            x = x + mha
            x = x + self._ff_block(self.norm3(x))
        else:
            # SA, Add, Norm, MHA, Add, Norm, FF, Add, Norm.
            sa, sa_weights = self._sa_block(
                x, tgt_mask, tgt_key_padding_mask, tgt_is_causal,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
            x = self.norm1(x + sa)

            mha, mha_weights = self._mha_block(
                x, memory, memory_mask, memory_key_padding_mask,
                memory_is_causal,
                # New args:
                return_weights=return_weights, average_heads=average_heads)
            x = self.norm2(x + mha)
            x = self.norm3(x + self._ff_block(x))

        return x, mha_weights, sa_weights

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  is_causal: bool = False,
                  # New args:
                  return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        x, weights = self.self_attn(
            x, x, x,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            # Modified args:
            need_weights=return_weights, average_attn_weights=average_heads)

        return self.dropout1(x), weights

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor],
                   is_causal: bool = False,
                   # New args:
                   return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Can now use need_weight = True.
        """
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask, key_padding_mask=key_padding_mask,
            is_causal=is_causal,
            # Modified args:
            need_weights=return_weights, average_attn_weights=average_heads)

        if return_weights:
            x, weights = x
        else:
            weights = None

        return self.dropout2(x[0]), weights
