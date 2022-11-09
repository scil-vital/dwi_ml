# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.experiment_utils.prints import format_dict_to_str

from dwi_ml.models.projects.transformers import (
    OriginalTransformerModel, TransformerSrcAndTgtModel)


def _create_batch():
    logging.debug("Creating batch: 2 streamlines, the first has 4 points "
                  "and the second, 3. Input: 4 features per point.")

    # dwi1 : data for the 3 first points
    flattened_dwi1 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.],
                      [60., 62., 62., 63.]]
    streamline1 = np.asarray([[0.1, 0.2, 0.3],
                              [1.1, 1.2, 1.3],
                              [2.1, 2.2, 2.3],
                              [3.1, 3.2, 3.3]])

    # dwi2 : data for the 2 first points
    flattened_dwi2 = [[10., 11., 12., 13.],
                      [50., 51., 52., 53.]]
    streamline2 = np.asarray([[10.1, 10.2, 10.3],
                              [11.1, 11.2, 11.3],
                              [12.1, 12.2, 12.3]])

    batch_x = [torch.Tensor(flattened_dwi1), torch.Tensor(flattened_dwi2)]
    batch_s = [streamline1, streamline2]

    return batch_x, batch_s


def test_models():
    batch_x, batch_streamlines = _create_batch()

    logging.debug("\n\nOriginal model!\n"
                  "-----------------------------")
    model = OriginalTransformerModel('test', nb_features=4,
                                     d_model=8, max_len=15,
                                     log_level='DEBUG',
                                     # Using defaults from script
                                     nb_previous_dirs=0,
                                     prev_dirs_embedding_size=None,
                                     prev_dirs_embedding_key=None,
                                     normalize_prev_dirs=True,
                                     positional_encoding_key='sinusoidal',
                                     embedding_key_x='nn_embedding',
                                     embedding_key_t='nn_embedding',
                                     ffnn_hidden_size=None,
                                     nheads=8, dropout_rate=0.1,
                                     activation='relu',
                                     n_layers_e=6, n_layers_d=6,
                                     dg_key='cosine-regression', dg_args=None,
                                     normalize_targets=True,
                                     neighborhood_type=None,
                                     neighborhood_radius=None)

    logging.info("Transformer original model final parameters:" +
                 format_dict_to_str(model.params_for_json_prints))

    # Testing forward.
    output = model(batch_x, batch_streamlines)

    assert len(output) == 2, "Model output should contain 2 streamlines"
    assert output[0].shape[1] == 3, "Model should output 3D coordinates"

    # Note. output[0].shape[0] ==> Depends if we unpad sequences.

    logging.debug("\n\nSource and target model!\n"
                  "-----------------------------")
    model = TransformerSrcAndTgtModel('test', nb_features=4,
                                      d_model=8, max_len=15,
                                      log_level='DEBUG',
                                      # Using defaults from script
                                      nb_previous_dirs=0,
                                      prev_dirs_embedding_size=None,
                                      prev_dirs_embedding_key=None,
                                      normalize_prev_dirs=True,
                                      positional_encoding_key='sinusoidal',
                                      embedding_key_x='nn_embedding',
                                      embedding_key_t='nn_embedding',
                                      ffnn_hidden_size=None,
                                      nheads=8, dropout_rate=0.1,
                                      activation='relu', n_layers_d=6,
                                      dg_key='cosine-regression', dg_args=None,
                                      normalize_targets=True,
                                      neighborhood_type=None,
                                      neighborhood_radius=None)


    logging.info("Transformer Src and Tgt model final parameters:" +
                 format_dict_to_str(model.params_for_json_prints))

    # Testing forward.
    output = model(batch_x, batch_streamlines)

    assert len(output) == 2, "Model output should contain 2 streamlines"
    assert output[0].shape[1] == 3, "Model should output 3D coordinates"


if __name__ == '__main__':
    logging.basicConfig(level='DEBUG')
    test_models()
