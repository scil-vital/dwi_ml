#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math

import numpy as np
from dwi_ml.models.projects.positional_encoding import (
    RelationalSinusoidalPosEncoding,
    SinusoidalPositionalEncoding)


def test_positional_encoding():
    d_model = 4
    max_len = 10
    dropout_rate = 0
    model = SinusoidalPositionalEncoding(d_model, dropout_rate, max_len)

    # Alternate computation:
    pos_emb2 = np.zeros((max_len, d_model))
    for index in range(0, d_model, 2):
        div_term = 10000.0 ** (index / d_model)
        pos_emb2[:, index] = np.array(
            [math.sin(pos / div_term) for pos in range(max_len)])
        if index + 1 < d_model:
            pos_emb2[:, index + 1] = np.array(
                [math.cos(pos / div_term) for pos in range(max_len)])

    assert np.allclose(np.asarray(model.pos_emb), np.asarray([pos_emb2]))
