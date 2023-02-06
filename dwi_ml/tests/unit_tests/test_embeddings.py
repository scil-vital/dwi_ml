#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging

import torch
from torch.nn.utils.rnn import pack_sequence

from dwi_ml.models.embeddings_on_tensors import \
    keys_to_embeddings as keys_on_tensors


def _verify_all_outputs(input_data, keys_to_embeddings, nb_features):
    logging.debug("Input is:\n{}".format(input_data))

    logging.debug('    Testing identity embedding...')
    cls = keys_to_embeddings['no_embedding']
    model = cls(input_size=nb_features, output_size=nb_features)
    output = model(input_data)
    logging.debug('    ==> Should return itself. Output is:\n{}'
                  .format(output))
    if isinstance(output, torch.Tensor):
        assert torch.equal(input_data, output)
    else:
        assert torch.equal(input_data.data, output.data)

    logging.debug('    Testing neural network embedding, ...')
    cls = keys_to_embeddings['nn_embedding']
    model = cls(input_size=nb_features, output_size=8)
    output = model(input_data)
    logging.debug('    ==> Should return output of size 8. Result is:\n{}'
                  .format(output))
    if isinstance(output, torch.Tensor):
        nb_outputs, size_outputs = output.size()
    else:
        nb_outputs, size_outputs = output.data.size()
    assert size_outputs == 8


def test_embeddings():
    logging.getLogger().setLevel(level='DEBUG')

    logging.debug("Unit test: embeddings on tensors")

    # Two inputs with 3 features
    tensor_a = torch.as_tensor([[0.0, 1.0, 2.2],
                                [10.3, 11.4, 12.5]])
    _verify_all_outputs(tensor_a, keys_on_tensors, nb_features=3)


if __name__ == '__main__':
    test_embeddings()
