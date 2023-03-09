# -*- coding: utf-8 -*-
import logging

from torch import Tensor, isnan, set_printoptions

from dwi_ml.models.projects.transforming_tractography import (
    OriginalTransformerModel, TransformerSrcAndTgtModel)


def _create_batch():
    logging.debug("Creating batch: 2 streamlines, the first has 4 points "
                  "and the second, 3. Input: 4 features per point.")

    # dwi1 : data for the 3 first points
    flattened_dwi1 = Tensor([[10., 11., 12., 13.],
                             [50., 51., 52., 53.],
                             [60., 62., 62., 63.]])
    streamline1 = Tensor([[0.1, 0.2, 0.3],
                          [1.1, 1.2, 1.3],
                          [2.1, 2.2, 2.3],
                          [3.1, 3.2, 3.3]])

    # dwi2 : data for the 2 first points
    flattened_dwi2 = Tensor([[10., 11., 12., 13.],
                             [50., 51., 52., 53.]])
    streamline2 = Tensor([[10.1, 10.2, 10.3],
                          [11.1, 11.2, 11.3],
                          [12.1, 12.2, 12.3]])

    batch_x_training = [flattened_dwi1, flattened_dwi2]
    batch_s_training = [streamline1, streamline2]

    batch_x_tracking = [flattened_dwi1[0:1, :], flattened_dwi2[0:1, :]]
    batch_s_tracking = [streamline1[0:1, :], streamline2[0:1, :]]

    return (batch_x_training, batch_x_tracking,
            batch_s_training, batch_s_tracking)


def test_models():
    (batch_x_training, batch_x_tracking,
     batch_s_training, batch_s_tracking) = _create_batch()
    total_nb_points_training = sum([len(s) - 1 for s in batch_s_training])
    nb_streamlines = len(batch_x_training)

    logging.debug("\n\nOriginal model!\n"
                  "-----------------------------")
    model = OriginalTransformerModel(experiment_name='test', nb_features=4,
                                     d_model=4, max_len=5,
                                     log_level='DEBUG',
                                     # Using defaults from script
                                     sphere_to_convert_input_dirs=None,
                                     sos_type_forward='as_label',
                                     positional_encoding_key='sinusoidal',
                                     embedding_key_x='nn_embedding',
                                     embedding_key_t='nn_embedding',
                                     ffnn_hidden_size=None,
                                     nheads=1, dropout_rate=0,
                                     activation='relu', norm_first=False,
                                     n_layers_e=1, n_layers_d=1,
                                     dg_key='cosine-regression', dg_args=None,
                                     neighborhood_type=None,
                                     neighborhood_radius=None)

    logging.debug("\n****** Training")
    # Testing forward. (Batch size = 2)
    output, weights = model(batch_x_training, batch_s_training,
                            return_weights=True)
    assert output.shape[0] == total_nb_points_training
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert len(weights) == 3  # Should get weights for encoder self-attention,
    #  decoder self-attention + multi-head attention.
    for weight in weights:
        assert weight is not None
    assert not isnan(output[0, 0])

    # Testing tracking
    logging.debug("\n****** Tracking")
    output = model(batch_x_tracking, batch_s_tracking, is_tracking=True)
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])

    # Note. output[0].shape[0] ==> Depends if we unpad sequences.

    logging.debug("\n\nSource and target model!\n"
                  "-----------------------------")
    model = TransformerSrcAndTgtModel(experiment_name='test', nb_features=4,
                                      d_model=4, max_len=5,
                                      log_level='DEBUG',
                                      # Using defaults from script
                                      sphere_to_convert_input_dirs=None,
                                      sos_type_forward='as_label',
                                      positional_encoding_key='sinusoidal',
                                      embedding_key_x='nn_embedding',
                                      embedding_key_t='nn_embedding',
                                      ffnn_hidden_size=None,
                                      nheads=1, dropout_rate=0,
                                      activation='relu', norm_first=False,
                                      n_layers_d=1,
                                      dg_key='cosine-regression', dg_args=None,
                                      neighborhood_type=None,
                                      neighborhood_radius=None)

    # Testing forward.
    logging.debug("\n****** Training")
    output, weights = model(batch_x_training, batch_s_training,
                            return_weights=True)
    assert output.shape[0] == total_nb_points_training
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert len(weights) == 1  # Should get weights for encoder self-attention
    assert not isnan(output[0, 0])
    for weight in weights:
        assert weight is not None

    # Testing tracking
    logging.debug("\n****** Tracking")
    output = model(batch_x_tracking, batch_s_tracking, is_tracking=True)
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    set_printoptions(precision=3, sci_mode=False)
    test_models()
