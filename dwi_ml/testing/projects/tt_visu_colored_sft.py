# -*- coding: utf-8 -*-
from dipy.io.stateful_tractogram import StatefulTractogram


def add_attention_as_dpp(sft: StatefulTractogram, attention_per_line, lengths,
                         attention_name):
    """
    Adds the attention's value to the data per point.

    attention_per_line: List of length nb layer.
        Each is: List[np.array] of length nb lines
    """
    nb_layers = len(attention_per_line[0])
    nb_heads = attention_per_line[0][0].shape[0]

    remaining_streamlines = sft.streamlines
    whole_sft = None

    import numpy as np
    for i, attline in enumerate(attention_per_line):
        for layer in range(nb_layers):
            print("   Line ", i, "of shape", lengths[i], len(sft.streamlines[i]))
            assert np.array_equal(attline[layer].shape, [nb_heads, lengths[i], lengths[i]]) , \
            "Expecting shape {} for line {} layer {} but got {}" \
            .format([nb_heads, lengths[i], lengths[i]], i, layer, attline[layer].shape)

    # Starting current point at length 2. At length 1, we know that it only
    # looked at the first point.
    for current_point in range(1, max(lengths)):
        # The nth point of each streamline, if long enough
        attention_per_line = [a for a, s in
                              zip(attention_per_line, remaining_streamlines)
                              if len(s) > current_point]
        remaining_streamlines = [s for s in remaining_streamlines
                                 if len(s) > current_point]

        # Saving first part of streamlines, up to current_point.
        # At current_point: which point did we look at?
        # Saving many ddp key for these streamlines: per layer, per head.
        tmp_sft = sft.from_sft([s[0:current_point]
                               for s in remaining_streamlines], sft)
        print("Adding the first {} points of the remaining {} streamlines"
              .format(current_point+1, len(remaining_streamlines)))
        for layer in range(nb_layers):
            for head in range(nb_heads):
                suffix = "_layer{}_head{}".format(layer, head)
                print("   Adding data per point: ", attention_name + suffix)
                ddp = [a[layer][head, current_point, :current_point]
                       for a in attention_per_line]
                tmp_sft.data_per_point[attention_name + suffix] = ddp

        if whole_sft is None:
            whole_sft = tmp_sft
        else:
            whole_sft = whole_sft + tmp_sft

    print("Total: {} streamlines".format(len(whole_sft)))

    return whole_sft
