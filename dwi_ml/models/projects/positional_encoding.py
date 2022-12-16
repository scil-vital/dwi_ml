# -*- coding: utf-8 -*-

"""
Positional embeddings:
    - SinusoidalPosEmbedding
    - RelationalSinusoidalPosEmbedding
"""
import math
import torch


class AbstractPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 2048):
        """
        Positional incoding for the Transformer model.

        Params
        ------
        d_model: int, size of the model
        dropout_rate: float
        max_len: int, maximum length of the sequences in input to the
            Transformer.
        """
        super().__init__()
        self.d_model = d_model
        self.dropout_rate = dropout_rate
        self.max_len = max_len

        self.dropout = None
        if self.dropout_rate > 0:
            self.dropout = torch.nn.Dropout(p=dropout_rate)


class SinusoidalPositionalEncoding(AbstractPositionalEncoding):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the paper
    Attention is all you need.

    Copied from
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model: int, dropout_rate, max_len: int = 2048):
        super().__init__(d_model, dropout_rate, max_len)

        assert d_model > 3, "Current computation of the Sinusoidal " \
                            "Positional encoding requires a d_model of size " \
                            "> 3."

        # Compute the sinusoidal embedding parameter
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))
        pos_emb = torch.zeros(1, max_len, d_model)
        pos_emb[0, :, 0::2] = torch.sin(position * div_term)
        pos_emb[0, :, 1::2] = torch.cos(position * div_term)

        # pos_emb is a parameter, but not learned. We don't want the optimizer
        # to update this. We could do self.pos_emb = pos_emb.
        # However, it is big enough that we would like it to be moved to GPU or
        # CPU whenever the module is. That is the use of a "buffer".
        self.register_buffer('pos_emb', pos_emb)

    def forward(self, x) -> torch.Tensor:
        """
        Args:
            x: Tensor. shape: [batch_size, seq_len, d_model]
        """
        # Important. Can't use +=. Inplace operation, backward propagation
        # would fail.
        x = x + self.pos_emb

        return x


class RelationalSinusoidalPosEncoding(AbstractPositionalEncoding):
    """
    Creates the vector for positional embedding that can be added to the data
    embedding as first layer of a transformer. Such as described in the music
    paper [ref] to replace the sinusoidal version.

    [ref]
    """
    def __init__(self, d_model, dropout_rate, max_len):
        super().__init__(d_model, dropout_rate, max_len)
        # toDo
        raise NotImplementedError


keys_to_positional_encodings = {
    'sinusoidal': SinusoidalPositionalEncoding,
    'relational': RelationalSinusoidalPosEncoding,
}
