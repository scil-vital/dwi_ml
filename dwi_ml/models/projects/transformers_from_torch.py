# -*- coding: utf-8 -*-
"""
We could use this for visu:
https://github.com/jessevig/bertviz

"""

from typing import Optional

import torch
from torch import Tensor, nan_to_num
from torch.nn import (Transformer,
                      TransformerDecoderLayer, TransformerDecoder,
                      TransformerEncoderLayer, TransformerEncoder)

# Shapes:
#  input: [nb_streamlines, max_len, d_model]
#  target: [nb_streamlines, max_len, d_model]
#  - Encoder:
#  --- Attention: (torch.nn.functionnal.multi_head_attention_forward):
#  n := nb streamlines * nb heads
#  ------ Combined masks future + padding: [n, max_len, max_len].
#         Contains -inf (or bool).
#  ------ After (Q*K), scaled, masked, softmax: [n, max_len, max_len].
#         Contains zeros where masked.
#  ------ After * V: [nb_streamlines, max_len, d_model].
#         Does not contain zeros anymore, that's ok.
#  ------ Output (called 'memory'): [nb_streamlines, max_len, d_model].
#  - Decoder: Similarly, output:
#      [nb streamlines, max_len, d_model]
#      Important. During softmax, during the self-attention, for the
#      first position, we get
#      softmax(-inf, -inf, -inf) = [nan, nan, nan].
#      Which are replaced by 0 in our modified Decoder.


class TransformerGetWeights(Transformer):
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
        is_batched = src.dim() == 3
        if not self.batch_first and src.size(1) != tgt.size(1) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        elif self.batch_first and src.size(0) != tgt.size(0) and is_batched:
            raise RuntimeError("the batch number of src and tgt must be equal")
        if src.size(-1) != self.d_model or tgt.size(-1) != self.d_model:
            raise RuntimeError("the feature number of src and tgt must be equal to d_model")

        memory, sa_weights_encoder = self.encoder(
            src, mask=src_mask, src_key_padding_mask=src_key_padding_mask,
            return_weights=return_weights, average_heads=average_heads)
        output, sa_weights_decoder, mha_weights = self.decoder(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
            return_weights=return_weights, average_heads=average_heads)

        return output, sa_weights_encoder, sa_weights_decoder, mha_weights


class TransformerEncoderGetWeights(TransformerEncoder):
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
            if _skpm_dtype != torch.bool and not torch.is_floating_point(src_key_padding_mask):
                raise AssertionError("only bool and floating types of key_padding_mask are supported")
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


class TransformerDecoderGetWeights(TransformerDecoder):
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


class TransformerEncoderLayerGetWeightsNoNaN(TransformerEncoderLayer):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                return_weights=False, average_heads=False):
        """
        Copy-pasted from torch. Now returns weights.
        Erased all the fast-track checks.
        """
        x = src
        if self.norm_first:
            sa, sa_weights = self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            sa = nan_to_num(sa)
            x = x + sa
            x = x + self._ff_block(self.norm2(x))
        else:
            sa, sa_weights = self._sa_block(x, src_mask, src_key_padding_mask)
            sa = nan_to_num(sa)
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
                                need_weights=False)
        if return_weights:
            x, weights = output
        else:
            x, weights = output, None

        return self.dropout1(x[0]), weights


class TransformerDecoderLayerGetWeightsNoNaN(TransformerDecoderLayer):
    """
    Decoder Layer, in the case where we do not have a start of sequence (SOS)
    token, and our mask contains only -inf for the first position. Output of
    self-attention becomes nan after the softmax step. Setting to 0.

    Also, now returning attention weights.
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

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
            sa, sa_weights = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            sa = nan_to_num(sa)
            x = x + sa
            mha, mha_weights = self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + mha
            x = x + self._ff_block(self.norm3(x))
        else:
            sa, sa_weights = self._sa_block(x, tgt_mask, tgt_key_padding_mask)
            sa = nan_to_num(sa)
            x = self.norm1(x + sa)
            mha, mha_weights = self._mha_block(x, memory, memory_mask, memory_key_padding_mask)
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

        if return_weights:
            x, weights = output
        else:
            x, weights = output, None

        return self.dropout1(x[0]), weights

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
