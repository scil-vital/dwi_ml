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
    IMAGE_SHAPE = np.asarray([4, 5, 6])

    # Two inputs with 3 features
    input_data = torch.rand([BATCH_SIZE, NB_FEATURES_IN])
    logging.debug("Input is:\n{}".format(input_data))

    # 1. No embedding
    logging.debug('---------------------------------')
    logging.debug('    Testing identity embedding...')
    cls = keys_to_embeddings['no_embedding']
    model = cls(nb_features_in=NB_FEATURES_IN, nb_features_out=NB_FEATURES_IN)
    output = model(input_data)
    logging.debug('    ==> Should return itself. Output is:\n{}'.format(output))
    assert torch.equal(input_data, output)

    # 2. NN embedding
    logging.debug('---------------------------------')
    logging.debug('    Testing neural network embedding, ...')
    cls = keys_to_embeddings['nn_embedding']
    model = cls(nb_features_in=NB_FEATURES_IN, nb_features_out=NB_FEATURES_OUT)
    output = model(input_data)
    logging.debug('    ==> Should return output of size {}. Result is:\n{}'
                  .format(NB_FEATURES_OUT, output.shape))
    nb_outputs, size_outputs = output.size()
    assert size_outputs == NB_FEATURES_OUT

    # 3. CNN embedding
    logging.debug('---------------------------------')
    logging.debug('    Testing CNN embedding, ...')
    input_data = torch.rand([BATCH_SIZE, *IMAGE_SHAPE, NB_FEATURES_IN])
    logging.debug("3D input: of shape {}".format(input_data.shape))

    cls = keys_to_embeddings['cnn_embedding']
    model = cls(nb_features_in=NB_FEATURES_IN, nb_features_out=NB_FEATURES_OUT,
                kernel_size=3, image_shape=IMAGE_SHAPE)

    # New input data: 3D image
    output = model(input_data)
    logging.debug('    ==> With kernel_size 3 (no padding, stride 1), should '
                  'return output \nof size dim - 2 on each side, for a total '
                  'of {}. Computed expected shape is {}. \nResulting flattened '
                  'shape should be {}: {}'
                  .format(IMAGE_SHAPE - 2, model.out_image_shape,
                          np.prod(IMAGE_SHAPE - 2), output.shape[1]))
    assert np.array_equal(IMAGE_SHAPE - 2, model.out_image_shape)
    assert output.shape[-1] == np.prod(IMAGE_SHAPE - 2) * NB_FEATURES_OUT


if __name__ == '__main__':
    test_embeddings()
