# -*- coding: utf-8 -*-
import logging

from torch.nn.utils.rnn import pack_sequence

from dwi_ml.models.stacked_rnn import StackedRNN, ADD_SKIP_TO_OUTPUT
from dwi_ml.unit_tests.utils.data_and_models_for_tests import \
    create_test_batch_2lines_4features

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



if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')

    print("---------------------------------------")
    print("Stacked RNN")
    print("---------------------------------------")
    test_stacked_rnn()

