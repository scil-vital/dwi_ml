# -*- coding: utf-8 -*-
import logging

from torch import isnan, set_printoptions

from dwi_ml.models.projects.transforming_tractography import (
    OriginalTransformerModel, TransformerSrcAndTgtModel)
from dwi_ml.unit_tests.utils.data_and_models_for_tests import create_test_batch

(batch_x_training, batch_x_tracking,
 batch_s_training, batch_s_tracking) = create_test_batch()
total_nb_points_training = sum([len(s) - 1 for s in batch_s_training])
nb_streamlines = len(batch_x_training)


def _prepare_original_model():
    # Using defaults from script
    model = OriginalTransformerModel(
        experiment_name='test', step_size=0.5, compress=None,
        nb_features=4, d_model=4, max_len=5,
        log_level='DEBUG', positional_encoding_key='sinusoidal',
        token_type='as_label', embedding_key_x='nn_embedding',
        embedding_key_t='nn_embedding', ffnn_hidden_size=None, nheads=1,
        dropout_rate=0., activation='relu', norm_first=False, n_layers_e=1,
        n_layers_d=1, dg_key='cosine-regression', dg_args=None,
        neighborhood_type=None, neighborhood_radius=None)
    return model


def _prepare_ttst_model():
    model = TransformerSrcAndTgtModel(
        experiment_name='test',  step_size=0.5, compress=None,
        nb_features=4, d_model=4, max_len=5,
        log_level='DEBUG', token_type='repulsion100',
        positional_encoding_key='sinusoidal', embedding_key_x='nn_embedding',
        embedding_key_t='nn_embedding', ffnn_hidden_size=None, nheads=1,
        dropout_rate=0., activation='relu', norm_first=False, n_layers_d=1,
        dg_key='cosine-regression', dg_args=None, neighborhood_type=None,
        neighborhood_radius=None)
    return model


def _run_original_model(model):
    logging.debug("\n****** Training")
    model.set_context('training')
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
    model.set_context('tracking')
    output = model(batch_x_tracking, batch_s_tracking)
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])

    # Note. output[0].shape[0] ==> Depends if we unpad sequences.


def _run_ttst_model(model):
    # Testing forward.
    logging.debug("\n****** Training")
    model.set_context('training')
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
    model.set_context('tracking')
    output = model(batch_x_tracking, batch_s_tracking)
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])


def test_models():
    logging.debug("\n\nOriginal model!\n"
                  "-----------------------------")

    model = _prepare_original_model()
    _run_original_model(model)

    # Note. output[0].shape[0] ==> Depends if we unpad sequences.

    logging.debug("\n\nSource and target model!\n"
                  "-----------------------------")

    model = _prepare_ttst_model()
    _run_ttst_model(model)


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    set_printoptions(precision=3, sci_mode=False)
    test_models()
