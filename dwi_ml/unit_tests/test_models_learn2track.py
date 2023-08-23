# -*- coding: utf-8 -*-
import logging

from torch.nn.utils.rnn import pack_sequence

from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.stacked_rnn import StackedRNN, ADD_SKIP_TO_OUTPUT
from dwi_ml.unit_tests.utils.data_and_models_for_tests import create_test_batch_2lines_4features

batch_x, _, batch_s, _ = create_test_batch_2lines_4features()


def test_stacked_rnn():
    batch_x_packed = pack_sequence(batch_x, enforce_sorted=False)

    model = StackedRNN('gru', input_size=4, layer_sizes=[3, 3],
                       use_skip_connection=True,
                       use_layer_normalization=True, dropout=0.4)

    # Model's logger level can be set by using the logger's name.
    logger = logging.getLogger('model_logger')
    logger.setLevel('DEBUG')

    # Testing forward.
    output, _hidden_state = model(batch_x_packed)

    assert len(output) == 5  # Total number of points.

    if ADD_SKIP_TO_OUTPUT:
        assert output.shape[1] == 10  # 4 + 3 + 3 with skip connections
    else:
        assert output.shape[1] == 6  # 3 + 3 with skip connections


def test_learn2track():
    model = Learn2TrackModel('test', step_size=0.5, compress_lines=False,
                             nb_features=4, rnn_layer_sizes=[3, 3],
                             log_level='DEBUG',
                             # Using default from script:
                             nb_previous_dirs=0, prev_dirs_embedded_size=None,
                             prev_dirs_embedding_key=None,
                             normalize_prev_dirs=True,
                             input_embedding_key='nn_embedding',
                             input_embedded_size=None, kernel_size=None,
                             nb_cnn_filters=None,
                             rnn_key='lstm', use_skip_connection=True,
                             use_layer_normalization=True, dropout=0.,
                             start_from_copy_prev=False,
                             dg_key='cosine-regression', dg_args=None,
                             neighborhood_type=None, neighborhood_radius=None)

    logging.info("Learn2track model final parameters:" +
                 format_dict_to_str(model.params_for_checkpoint))

    # Testing forward. No previous dirs
    model.set_context('training')
    model(batch_x, batch_s)


def test_learn2track_cnn():
    model = Learn2TrackModel('test', step_size=0.5, compress_lines=False,
                             nb_features=4, rnn_layer_sizes=[3, 3],
                             log_level='DEBUG',
                             # Using default from script:
                             nb_previous_dirs=0, prev_dirs_embedded_size=None,
                             prev_dirs_embedding_key=None,
                             normalize_prev_dirs=True,
                             input_embedding_key='cnn_embedding',
                             nb_cnn_filters=4, kernel_size=3,
                             input_embedded_size=None,
                             rnn_key='lstm', use_skip_connection=True,
                             use_layer_normalization=True, dropout=0.,
                             start_from_copy_prev=False,
                             dg_key='cosine-regression', dg_args=None,
                             neighborhood_type='grid', neighborhood_radius=1,
                             neighborhood_resolution=1)

    logging.info("Learn2track model final parameters:" +
                 format_dict_to_str(model.params_for_checkpoint))

    # Testing forward. No previous dirs
    model.set_context('training')
    # pretending that we have 27 neighbors.
    for i, line_input in enumerate(batch_x):
        batch_x[i] = line_input.repeat(1, 27)
    model(batch_x, batch_s)


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')

    print("---------------------------------------")
    print("Stacked RNN")
    print("---------------------------------------")
    test_stacked_rnn()

    print("\n---------------------------------------")
    print("Model Learn2track")
    print("---------------------------------------")
    test_learn2track()

    print("\n---------------------------------------")
    print("Model Learn2track with CNN input embedding")
    print("---------------------------------------")
    test_learn2track_cnn()
