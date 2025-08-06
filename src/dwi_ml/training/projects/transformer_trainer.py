# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch

from dwi_ml.tracking.io_utils import prepare_tracking_mask
from dwi_ml.tracking.propagation import propagate_multiple_lines

from dwi_ml.training.trainers_withGV import \
    DWIMLTrainerForTrackingOneInput


class TransformerTrainer(DWIMLTrainerForTrackingOneInput):
    def __init__(self, **kwargs):
        """
        See Super for parameter description. No additional parameters here.
        """
        super().__init__(**kwargs)

    def propagate_multiple_lines(self, lines: List[torch.Tensor],
                                 ids_per_subj):
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        # Getting the first inputs
        tmp_lines = [line[:-1, :] for line in lines]
        batch_inputs = self.batch_loader.load_batch_inputs(tmp_lines,
                                                           ids_per_subj)
        del tmp_lines

        def update_memory_after_removing_lines(can_continue: np.ndarray, __):
            nonlocal batch_inputs
            batch_inputs = [inp for i, inp in enumerate(batch_inputs) if
                            can_continue[i]]

        def get_dirs_at_last_pos(_lines: List[torch.Tensor], n_last_pos):
            nonlocal batch_inputs
            n_last_pos = [pos[None, :] for pos in n_last_pos]
            latest_inputs = self.batch_loader.load_batch_inputs(
                n_last_pos, ids_per_subj)
            batch_inputs = [torch.vstack((first, last)) for first, last in
                            zip(batch_inputs, latest_inputs)]

            model_outputs = self.model(batch_inputs, _lines)
            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        final_lines = []
        for subj_idx, line_idx in ids_per_subj.items():

            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r'
                           ) as hdf_handle:
                subj_id = self.batch_loader.context_subset.subjects[subj_idx]
                logging.debug("Loading subj {} ({})'s tracking mask."
                              .format(subj_idx, subj_id))
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest')
                tracking_mask.move_to(self.device)

            final_lines.extend(propagate_multiple_lines(
                lines[line_idx], update_memory_after_removing_lines,
                get_dirs_at_last_pos, theta=theta,
                step_size=self.model.step_size,
                verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))

        return final_lines
