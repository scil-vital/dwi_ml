#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch

from dwi_ml.models.embeddings import keys_to_embeddings


def test_embeddings():
    logging.getLogger().setLevel(level='DEBUG')

    BATCH_SIZE = 2
    NB_FEATURES_IN = 3
    NB_FEATURES_OUT = 7

    # Two inputs with 3 features
    input_data = torch.rand([BATCH_SIZE, NB_FEATURES_IN])

    # 1. No embedding
    logging.debug('---------------------------------')
    logging.debug('    Testing identity embedding...')
    logging.debug("    Input is: nb points x nb_features: {}"
                  .format(input_data.shape))
    cls = keys_to_embeddings['no_embedding']
    model = cls(nb_features_in=NB_FEATURES_IN)
    output = model(input_data)
    logging.debug('    ==> Should return itself.')
    assert torch.equal(input_data, output)

    # 2. NN embedding
    logging.debug('\n---------------------------------')
    logging.debug('    Testing neural network embedding, ...')
    logging.debug("    Input is: nb points x nb_features: {}"
                  .format(input_data.shape))
    cls = keys_to_embeddings['nn_embedding']
    model = cls(nb_features_in=NB_FEATURES_IN, nb_features_out=NB_FEATURES_OUT)
    output = model(input_data)
    logging.debug('    ==> Should return output of size {}. Result is: {}'
                  .format(NB_FEATURES_OUT, output.shape))
    nb_outputs, size_outputs = output.size()
    assert size_outputs == NB_FEATURES_OUT

    # 3. CNN embedding
    logging.debug('\n---------------------------------')
    logging.debug('    Testing CNN embedding, ...')

    # New input data: 3D image. Using 5x5x5 like neighborhood grid 2.
    IMAGE_SHAPE = np.asarray([5, 5, 5])
    input_data = torch.rand([BATCH_SIZE, *IMAGE_SHAPE, NB_FEATURES_IN])
    logging.debug("    3D input: of shape batch * 3d shape * nb_features: {}"
                  .format(input_data.shape))

    cls = keys_to_embeddings['cnn_embedding']

    logging.debug("     1. Testing kernel of size 5 = fits only once.")
    model = cls(nb_features_in=NB_FEATURES_IN, nb_filters=[NB_FEATURES_OUT],
                kernel_sizes=[5], image_shape=IMAGE_SHAPE)
    logging.debug('    ==> With kernel_size 3 (no padding, stride 1), should '
                  ' fit only once.')

    output = model(input_data)
    logging.debug("    Computed out shape: batch {} x flattened size {} "
                  "(shape {} x out features {}). Got out shape: {}"
                  .format(BATCH_SIZE, model.out_flattened_size,
                          model.out_image_shape, NB_FEATURES_OUT,
                          output.shape))

    assert output.shape[-1] == np.prod(IMAGE_SHAPE - 4) * NB_FEATURES_OUT
    assert output.shape[-1] == model.out_flattened_size
    assert np.array_equal(output.shape, [BATCH_SIZE, model.out_flattened_size])

    logging.debug("")
    logging.debug("     2. Testing kernel of size 3 = fits three times.")
    logging.debug('    ==> With kernel_size 3 (no padding, stride 1), should '
                  'return output of size dim - 2 on each side, for a total '
                  'of {}.'.format(IMAGE_SHAPE - 2))
    model = cls(nb_features_in=NB_FEATURES_IN, nb_filters=[NB_FEATURES_OUT],
                kernel_sizes=[3], image_shape=IMAGE_SHAPE)

    output = model(input_data)
    logging.debug("    Computed out shape: batch {} x flattened size {} "
                  "(shape {} x out features {}). Got out shape: {}"
                  .format(BATCH_SIZE, model.out_flattened_size,
                          model.out_image_shape, NB_FEATURES_OUT,
                          output.shape))

    assert np.array_equal(IMAGE_SHAPE - 2, model.out_image_shape)
    assert output.shape[-1] == model.out_flattened_size
    assert output.shape[-1] == np.prod(IMAGE_SHAPE - 2) * NB_FEATURES_OUT

    logging.debug("")
    logging.debug("     2. Testing 2 layers with kernel of size 3 = "
                  "fits three times on first layer, once one second.")
    model = cls(nb_features_in=NB_FEATURES_IN, nb_filters=[NB_FEATURES_OUT]*2,
                kernel_sizes=[3, 3], image_shape=IMAGE_SHAPE)
    output = model(input_data)

    assert output.shape[-1] == np.prod(IMAGE_SHAPE - 2 - 2) * NB_FEATURES_OUT
    assert output.shape[-1] == model.out_flattened_size
    assert np.array_equal(output.shape, [BATCH_SIZE, model.out_flattened_size])


if __name__ == '__main__':
    test_embeddings()
