# -*- coding: utf-8 -*-
from typing import List, Dict

import torch

from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput


class DWIMLBatchLoaderWithConnectivity(DWIMLBatchLoaderOneInput):
    def __init__(self, **kwargs):
        assert "hdf5 contains connectivity"
        super().__init__(**kwargs)

    def load_batch_connectivity_matrices(
            self, streamline_ids_per_subj: Dict[int, slice]):
        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        subjs = list(streamline_ids_per_subj.keys())
        nb_subjs = len(subjs)
        matrices = [None] * nb_subjs
        volume_sizes = [None] * nb_subjs
        downsampled_sizes = [None] * nb_subjs
        for i, subj in enumerate(subjs):
            # No cache for the sft data. Accessing it directly.
            # Note: If this is used through the dataloader, multiprocessing
            # is used. Each process will open a handle.
            subj_data = \
                self.context_subset.subjs_data_list.get_subj_with_handle(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # We could access it only at required index maybe. Loading the
            # whole matrix here.
            matrices[i], volume_sizes[i], downsampled_sizes[i] = \
                subj_sft_data.connectivity_matrix_and_info()

        return matrices, volume_sizes, downsampled_sizes
