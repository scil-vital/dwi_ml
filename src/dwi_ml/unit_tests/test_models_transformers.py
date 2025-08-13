# -*- coding: utf-8 -*-
import logging

from torch import isnan, set_printoptions

from dwi_ml.models.projects.transformer_models import (
    OriginalTransformerModel, TransformerSrcAndTgtModel, TransformerSrcOnlyModel)
from dwi_ml.unit_tests.utils.data_and_models_for_tests import create_test_batch_2lines_4features

(batch_x_various_lengths, batch_x_same_lengths,
 batch_s_various_lengths, batch_s_same_lengths) = create_test_batch_2lines_4features()
total_nb_points_training = sum([len(s) for s in batch_s_various_lengths])
nb_streamlines = len(batch_x_various_lengths)


def _prepare_original_model():
    # Nb features per point = 4.
    # Pretending that this is 1 + 3 coordinates.

    # Using defaults from script
    model = OriginalTransformerModel(
        experiment_name='test', step_size=0.5, compress_lines=None,
        nb_features_per_point=1, input_embedding_nn_out_size=4, max_len=5,
        log_level='DEBUG', positional_encoding_key='sinusoidal',
        sos_token_type='as_label', input_embedding_key='nn_embedding',
        target_embedding_key='nn_embedding', ffnn_hidden_size=None, nheads=1,
        dropout_rate=0., activation='relu', norm_first=False, n_layers_e=1,
        n_layers_d=1, dg_key='cosine-regression', dg_args=None,
        neighborhood_type=None, neighborhood_radius=None,
        input_embedding_cnn_nb_filters=None,
        input_embedding_cnn_kernel_size=None,
        add_raw_coords_to_input=False,
        add_relative_coords_to_input=True,
        start_from_copy_prev=False)
    return model


def _prepare_ttst_model():
    # neighborhood --> No. The number of features is fixed in our fake data

    model = TransformerSrcAndTgtModel(
        experiment_name='test',  step_size=0.5, compress_lines=None,
        nb_features_per_point=4, max_len=5,
        target_embedded_size=2,
        log_level='DEBUG', sos_token_type='repulsion100',
        positional_encoding_key='sinusoidal',
        input_embedding_key='nn_embedding',
        target_embedding_key='nn_embedding', ffnn_hidden_size=6, nheads=1,
        dropout_rate=0., activation='relu', norm_first=True, n_layers_e=1,
        dg_key='sphere-classification', dg_args={'add_eos': True},
        neighborhood_type=None, neighborhood_radius=None,
        input_embedding_nn_out_size=4,
        input_embedding_cnn_nb_filters=None,
        input_embedding_cnn_kernel_size=None,
        add_raw_coords_to_input=False,
        add_relative_coords_to_input=False,
        start_from_copy_prev=True)
    return model


def _prepare_tts_model():
    model = TransformerSrcOnlyModel(
        experiment_name='test',  step_size=0.5, compress_lines=None,
        nb_features_per_point=4, max_len=5, log_level='DEBUG',
        input_embedding_nn_out_size=4,
        positional_encoding_key='sinusoidal',
        input_embedding_key='nn_embedding',
        ffnn_hidden_size=None, nheads=1, dropout_rate=0., activation='relu',
        norm_first=False, n_layers_e=1, dg_key='cosine-regression',
        dg_args=None, neighborhood_type=None, neighborhood_radius=None,
        input_embedding_cnn_nb_filters=None,
        input_embedding_cnn_kernel_size=None,
        add_raw_coords_to_input=False,
        add_relative_coords_to_input=False)
    return model


def _run_original_model(model):
    logging.debug("\n****** Training")
    model.set_context('training')
    # Testing forward. (Batch size = 2)
    output = model(batch_x_various_lengths, batch_s_various_lengths)
    assert len(output) == nb_streamlines
    assert output[0].shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0][0, 0])

    # Testing forward during visu context: will return weights
    model.set_context('visu_weights')
    _, weights = model(batch_x_various_lengths, batch_s_various_lengths)
    assert len(weights) == 3  # Should get weights for encoder self-attention,
    #  decoder self-attention + multi-head attention.
    for weight in weights:
        assert weight is not None

    # Testing tracking
    # Using batch with same lengths: mimicking forward tracking.
    logging.debug("\n****** Tracking one step, forward tracking")
    model.set_context('tracking')
    output = model(batch_x_same_lengths, batch_s_same_lengths)
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])

    # Note. output[0].shape[0] ==> Depends on if we unpad sequences.


def _run_ttst_model(model):
    # Testing forward.
    logging.debug("\n****** Training")
    model.set_context('training')
    output = model(batch_x_various_lengths, batch_s_various_lengths)
    assert len(output) == nb_streamlines
    # Here, classification, output should be our sphere's size (724) + 1 for
    # EOS.
    assert output[0].shape[1] == 725
    assert not isnan(output[0][0, 0])

    # Testing forward during visu context: will return weights
    model.set_context('visu_weights')
    _, weights = model(batch_x_various_lengths, batch_s_various_lengths)
    assert len(weights) == 1  # Should get weights for encoder self-attention
    for weight in weights:
        assert weight is not None

    # Testing tracking
    # This time, using batch with various lengths: mimicking backward tracking.
    logging.debug("\n****** Tracking one step, backward tracking")
    model.set_context('tracking')
    output = model(batch_x_various_lengths, batch_s_various_lengths)

    # Output is not split.
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 725
    assert not isnan(output[0, 0])


def _run_tts_model(model):
    # Testing forward.
    logging.debug("\n****** Training")
    model.set_context('training')
    output = model(batch_x_various_lengths, None)
    assert len(output) == nb_streamlines
    assert output[0].shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0][0, 0])

    # Testing forward during visu context: will return weights
    model.set_context('visu_weights')
    _, weights = model(batch_x_various_lengths, None)
    assert len(weights) == 1  # Should get weights for encoder self-attention
    for weight in weights:
        assert weight is not None

    # Testing tracking
    logging.debug("\n****** Tracking one step")
    model.set_context('tracking')
    output = model(batch_x_same_lengths, batch_s_same_lengths)

    # Output is not split.
    assert output.shape[0] == nb_streamlines
    assert output.shape[1] == 3  # Here, regression, should output x, y, z
    assert not isnan(output[0, 0])


def test_models():
    logging.debug("\n\nOriginal model!\n"
                  "-----------------------------")
    model = _prepare_original_model()
    _run_original_model(model)

    logging.debug("\n\nSource and target model!\n"
                  "-----------------------------")
    model = _prepare_ttst_model()
    _run_ttst_model(model)

    logging.debug("\n\nSource only model!\n"
                  "-----------------------------")
    model = _prepare_tts_model()
    _run_tts_model(model)


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    set_printoptions(precision=3, sci_mode=False)
    test_models()
