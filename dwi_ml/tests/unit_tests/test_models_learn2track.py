# -*- coding: utf-8 -*-
import logging

from dwi_ml.experiment_utils.prints import format_dict_to_str
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.projects.learn2track_model import StackedRNN
from dwi_ml.tests.utils.data_and_models_for_tests import create_test_batch

batch_x, _, batch_s, _ = create_test_batch()


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
    assert output.shape[1] == 6  # 3 + 3 with skip connections


def test_learn2track():
    model = Learn2TrackModel('test', nb_features=4, rnn_layer_sizes=[3, 3],
                             log_level='DEBUG',
                             # Using default from script:
                             nb_previous_dirs=0, prev_dirs_embedding_size=None,
                             prev_dirs_embedding_key=None,
                             normalize_prev_dirs=True,
                             input_embedding_key='no_embedding',
                             input_embedding_size=None,
                             input_embedding_size_ratio=None,
                             rnn_key='lstm', use_skip_connection=True,
                             use_layer_normalization=True, dropout=0.,
                             dg_key='cosine-regression', dg_args=None,
                             neighborhood_type=None, neighborhood_radius=None)

    logging.info("Transformer original model final parameters:" +
                 format_dict_to_str(model.params_for_checkpoint))

    # Testing forward. No previous dirs
    model.set_context('training')
    model(batch_x, batch_s)


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_stacked_rnn()
    test_learn2track()
