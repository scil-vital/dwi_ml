# -*- coding: utf-8 -*-
import torch
from torch.nn.utils.rnn import PackedSequence


def unpack_sequence(packed_sequence: PackedSequence,
                    get_indices_only: bool = False):
    """
    How it works. Ex:
    streamline0 = torch.Tensor([[0,0,0], [1,1,1]])
    streamline1 = torch.Tensor([[2,2,2], [3,3,3], [4,4,4]])
    packed_sequence = torch.nn.utils.rnn.pack_sequence(
            [streamline0, streamline1], enforce_sorted = False)

    Then,
    packed_sequence.data:
        tensor([[2., 2., 2.],  ---> s1.1
                [0., 0., 0.],  ---> s0.1
                [3., 3., 3.],  ---> s1.2
                [1., 1., 1.],  ---> s0.2
                [4., 4., 4.]]) ---> s1.3
    packed_sequence.batch_sizes:
        tensor([2,2,1]) --->  means that the 2 first data are the first
                              timestep, 2 for the second timestep, 1 x a third
                              timestep.
    packed_sequence.unsorted_indices:
        tensor([1,0]) --->  shows that streamlines were inversed

    Returns:
    --------
    if get_indices_only:
        indices_per_streamline: List[List]
            The indices of each streamline in packed_input.data
    else:
        streamlines: List[List]
            The list of unpacked streamlines.
    """
    nb_streamlines = len(packed_sequence.unsorted_indices)

    indices_per_streamline = [None]*nb_streamlines
    undealed_batch_size = packed_sequence.batch_sizes

    i = 0
    for s in packed_sequence.unsorted_indices.tolist():
        current_streamline_length = len(undealed_batch_size)
        streamline_indices = \
            range(i, current_streamline_length*nb_streamlines + i,
                  nb_streamlines)
        indices_per_streamline[s] = streamline_indices

        undealed_batch_size = torch.sub(undealed_batch_size, 1)
        undealed_batch_size = undealed_batch_size[undealed_batch_size > 0]

        i += 1

    if get_indices_only:
        return indices_per_streamline
    else:
        streamlines = unpack_tensor_from_indices(packed_sequence.data,
                                                 indices_per_streamline)
        return streamlines


def unpack_tensor_from_indices(inputs, indices):
    """Separated from function above because it can be useful in the case
    were a packed_sequence.data has been modified (ex, through a neural
    network) but is not an instance of packed_sequence anymore."""
    outputs = [inputs[indices[i]] for i in range(len(indices))]
    return outputs
