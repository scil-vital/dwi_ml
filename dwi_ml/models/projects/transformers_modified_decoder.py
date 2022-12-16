# -*- coding: utf-8 -*-
import logging
from typing import Optional

import torch
from torch import Tensor
from torch.nn import TransformerDecoderLayer


class TransformerDecoderLayerNoSOS(TransformerDecoderLayer):
    """
    Decoder Layer, in the case where we do not have a start of sequence (SOS)
    token, and our mask contains only -inf for the first position. Output of
    self-attention becomes nan after the softmax step. Setting to 0.
    """
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)

    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.

        (COPY PASTED FROM TORCH, MODIFIED)
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x = torch.nan_to_num(x)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
            x = torch.nan_to_num(x)
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))

        return x

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor],
                   key_padding_mask: Optional[Tensor]) -> Tensor:
        """
        Copy-pasted from torch. Can now use need_weight = True.
        """
        x, weights = self.multihead_attn(x, mem, mem,
                                         attn_mask=attn_mask,
                                         key_padding_mask=key_padding_mask,
                                         need_weights=True)
        logging.warning("We can now save the weights to look at them! Yeah!"
                        "(todo)")

        x = x[0]
        return self.dropout2(x)
