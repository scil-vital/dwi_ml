# -*- coding: utf-8 -*-
import logging

import torch

from dwi_ml.data.processing.streamlines.data_augmentation import \
    resample_or_compress
from dwi_ml.models.projects.copy_previous_dirs import CopyPrevDirModel
from dwi_ml.testing.testers import Tester
from dwi_ml.testing.utils import prepare_dataset_one_subj


class TesterCopyPrevDir(Tester):
    def __init__(self, model: CopyPrevDirModel,
                 streamline_group, batch_size: int = None,
                 device: torch.device = None):
        super().__init__('no_experiment', model, batch_size, device)
        self.streamline_group = streamline_group

    @property
    def _volume_groups(self):
        return []

    def _prepare_inputs_at_pos(self, streamlines):
        return None

    def load_and_format_data(self, subj_id, hdf5_file, subset_name,
                             step_size, compress):
        # No checkpoint to find batch loader information. Requires additional
        # params compared to super.

        # 1. Load subject
        logging.info("Loading subject {} from hdf5.".format(subj_id))
        self.subset, self.subj_idx = prepare_dataset_one_subj(
            hdf5_file, subj_id, subset_name=subset_name, volume_groups=[],
            streamline_groups=[self.streamline_group], lazy=False,
            cache_size=None)

        # 2. Load SFT as in dataloader, except we don't loop on many subject,
        # we don't verify streamline ids (loading all), and we don't split /
        # reverse streamlines. But we resample / compress.
        logging.info("Loading its streamlines as SFT.")
        streamline_group_idx = self.subset.streamline_groups.index(
            self.streamline_group)
        subj_data = self.subset.subjs_data_list.get_subj_with_handle(self.subj_idx)
        subj_sft_data = subj_data.sft_data_list[streamline_group_idx]
        sft = subj_sft_data.as_sft()

        sft = resample_or_compress(sft, step_size, compress)
        sft.to_vox()
        sft.to_corner()

        return sft
