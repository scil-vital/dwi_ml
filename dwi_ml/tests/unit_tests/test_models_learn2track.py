# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.experiment_utils.prints import format_dict_to_str
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.models.projects.learn2track_model import Learn2TrackModel
from dwi_ml.models.projects.learn2track_model import StackedRNN


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

    batch_x = [torch.as_tensor(flattened_dwi1),
               torch.as_tensor(flattened_dwi2)]
    batch_s = [streamline1, streamline2]

    return batch_x, batch_s


def test_stacked_rnn():
    batch_x, _ = _create_batch()
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
    fake_x, fake_s = _create_batch()

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
                             normalize_targets=True,
                             neighborhood_type=None, neighborhood_radius=None)

    logging.info("Transformer original model final parameters:" +
                 format_dict_to_str(model.params_for_json_prints))

    # Testing forward. No previous dirs
    model(fake_x, fake_s)


if __name__ == '__main__':
    logging.getLogger().setLevel(level='DEBUG')
    test_stacked_rnn()
    test_learn2track()
