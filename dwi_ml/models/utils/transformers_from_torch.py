# -*- coding: utf-8 -*-
"""
Child classes of Torch Transformers. Changes are:

- Main transformer: Now returns the weights for visualisation + Added an option
  to decide if we want to share the linear weights for Q, K, V.
- Encoder: Idem
- Decoder: Idem
- EncoderLayer: Idem
- DecoderLayer: Idem

"""
from typing import Optional

import torch
from torch import Tensor
from torch.nn import (Transformer,
                      TransformerDecoderLayer, TransformerDecoder,
                      TransformerEncoderLayer, TransformerEncoder,
                      MultiheadAttention, Parameter)


class ModifiedTransformer(Transformer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor = None,
                tgt_mask: Tensor = None, memory_mask: Tensor = None,
                src_key_padding_mask: Tensor = None,
                tgt_key_padding_mask: Tensor = None,
                memory_key_padding_mask: Tensor = None,
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        memory, sa_weights_encoder = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            return_weights=return_weights, average_heads=average_heads)

        output, sa_weights_decoder, mha_weights = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_weights=return_weights, average_heads=average_heads)

        return output, sa_weights_encoder, sa_weights_decoder, mha_weights


class ModifiedTransformerEncoder(TransformerEncoder):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        Erased all the fast-path check: it is not used anyway if it is
        training, and if we supply both src_key_padding_mask and mask, which is
        our case.
        Layers must be TransformerEncoderLayerGetWeights layers.
        """
        if src_key_padding_mask is not None:
            _skpm_dtype = src_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not \
                    torch.is_floating_point(src_key_padding_mask):
                raise AssertionError("only bool and floating types of "
                                     "key_padding_mask are supported")
        output = src
        src_key_padding_mask_for_layers = src_key_padding_mask
        sa_weights = [None] * len(self.layers)

        for mod, i in zip(self.layers, range(len(self.layers))):
            output, sa_weights[i] = mod(
                output, src_mask=mask,
                src_key_padding_mask=src_key_padding_mask_for_layers,
                return_weights=return_weights, average_heads=average_heads)

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_weights


class ModifiedTransformerDecoder(TransformerDecoder):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, tgt: Tensor, memory: Tensor,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        Layers must be TransformerDecoderLayerGetWeightsNoSOS layers.
        """
        output = tgt
        mha_weights = [None]*len(self.layers)
        sa_weights = [None] * len(self.layers)

        for mod, i in zip(self.layers, range(len(self.layers))):
            output, mha_weights[i], sa_weights[i] = \
                mod(output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                    return_weights=return_weights,
                    average_heads=average_heads)

        if self.norm is not None:
            output = self.norm(output)

        return output, sa_weights, mha_weights


def do_not_share_linear_weights(attn: MultiheadAttention, d_model):
    """
    I added a request for this parameter to be accessible.
    https://github.com/pytorch/pytorch/issues/92990
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
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        Erased all the fast-track checks.
        """
        x = src
        if self.norm_first:
            # Norm, SA, Add, Norm, FF, Add
            sa, sa_weights = self._sa_block(self.norm1(x), src_mask,
                                            src_key_padding_mask,
                                            return_weights)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            # SA, Add, Norm, FF, Add, Norm
            sa, sa_weights = self._sa_block(x, src_mask, src_key_padding_mask,
                                            return_weights)
            x = self.norm1(x + sa)
            x = self.norm2(x + self._ff_block(x))

        return x, sa_weights

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  return_weights=False, average_heads=False):
        output = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=return_weights)
        x, weights = output  # if return_weights is False, weights is None

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
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights + converts nan to 0.
        Weights are None if return_weights is False.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        if self.norm_first:
            # Norm, SA, Add, Norm, MHA, Add, Norm, FF, Add
            sa, sa_weights = self._sa_block(self.norm1(x), tgt_mask,
                                            tgt_key_padding_mask,
                                            return_weights=return_weights)
            x = x + sa
            mha, mha_weights = self._mha_block(self.norm2(x), memory,
                                               memory_mask,
                                               memory_key_padding_mask,
                                               return_weights=return_weights)
            x = x + mha
            x = x + self._ff_block(self.norm3(x))
        else:
            # SA, Add, Norm, MHA, Add, Norm, FF, Add, Norm.
            sa, sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask,
                                            return_weights=return_weights)
            x = self.norm1(x + sa)

            mha, mha_weights = self._mha_block(x, memory, memory_mask,
                                               memory_key_padding_mask,
                                               return_weights=return_weights)
            x = self.norm2(x + mha)
            x = self.norm3(x + self._ff_block(x))

        return x, mha_weights, sa_weights

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor],
                  key_padding_mask: Optional[Tensor],
                  return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        """
        output = self.self_attn(x, x, x,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                need_weights=return_weights,
                                average_attn_weights=average_heads)

        x, weights = output  # If not return_weights, weights is None.

        return self.dropout1(x), weights

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor],
                   return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Can now use need_weight = True.
        """
        output = self.multihead_attn(x, mem, mem,
                                     attn_mask=attn_mask,
                                     key_padding_mask=key_padding_mask,
                                     need_weights=return_weights,
                                     average_attn_weights=average_heads)

        if return_weights:
            x, weights = output
        else:
            x, weights = output, None

        return self.dropout2(x[0]), weights
