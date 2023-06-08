# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch
from torch.nn import PairwiseDistance

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_triu_connectivity
from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.projects.utils import prepare_tracking_mask
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.monitoring import BatchHistoryMonitor
from dwi_ml.training.with_generation.batch_loader import \
    DWIMLBatchLoaderWithConnectivity

logger = logging.getLogger('train_logger')

# Emma tests in ISMRM: sphere of 50 mm of diameter (in MI-Brain, imagining a
# sphere encapsulated in a cube box of 50x50x50) around any point seems to
# englobe mostly acceptable streamlines.
# So half of it, a ray of 25 mm seems ok.
VERY_CLOSE_THRESHOLD = 15.0
ACCEPTABLE_THRESHOLD = 25.0
VERY_FAR_THRESHOLD = 40.0


class DWIMLTrainerForTrackingOneInput(DWIMLTrainerOneInput):
    model: ModelWithDirectionGetter
    batch_loader: DWIMLBatchLoaderWithConnectivity

    def __init__(self, add_a_tracking_validation_phase: bool = False,
                 tracking_phase_frequency: int = 5,
                 tracking_phase_nb_steps_init: int = 5,
                 tracking_phase_mask_group: str = None, *args, **kw):
        super().__init__(*args, **kw)

        self.add_a_tracking_validation_phase = add_a_tracking_validation_phase
        self.tracking_phase_frequency = tracking_phase_frequency
        self.tracking_phase_nb_steps_init = tracking_phase_nb_steps_init
        self.tracking_mask_group = tracking_phase_mask_group

        self.tracking_mask = None
        if add_a_tracking_validation_phase:
            # Right now, using any subject's, and supposing that they are all
            # in the same space. Else, code would need refactoring to allow
            # tracking on multiple subjects. Or we can loop on each subject.
            logging.warning("***************\n"
                            "CODE NEEDS REFACTORING. USING THE SAME TRACKING "
                            "MASK FOR ALL SUBJECTS.\n"
                            "***************\n")
            any_subj = self.batch_loader.dataset.training_set.subjects[0]
            if tracking_phase_mask_group is not None:
                with h5py.File(self.batch_loader.dataset.hdf5_file, 'r') \
                        as hdf_handle:
                    logging.info("Loading tracking mask.")
                    self.tracking_mask, _ = prepare_tracking_mask(
                        hdf_handle, tracking_phase_mask_group, subj_id=any_subj,
                        mask_interp='nearest')
                    self.tracking_mask.move_to(self.device)

        # -------- Monitors
        # At training time: only the one metric used for training.
        # At validation time: A lot of exploratory metrics monitors.

        # Percentage of streamlines inside a radius
        self.tracking_very_good_IS_monitor = BatchHistoryMonitor(
            'tracking_very_good_IS_monitor', weighted=True)
        self.tracking_acceptable_IS_monitor = BatchHistoryMonitor(
            'tracking_acceptable_IS_monitor', weighted=True)
        self.tracking_very_far_IS_monitor = BatchHistoryMonitor(
            'tracking_very_far_IS_monitor', weighted=True)

        # Point where the streamline starts diverging from "acceptable"
        self.tracking_valid_diverg_monitor = BatchHistoryMonitor(
            'tracking_valid_diverg_monitor', weighted=True)

        # Final distance from expected point
        self.tracking_mean_final_distance_monitor = BatchHistoryMonitor(
            'tracking_mean_final_distance_monitor', weighted=True)
        self.tracking_clipped_final_distance_monitor = BatchHistoryMonitor(
            'tracking_clipped_final_distance_monitor', weighted=True)

        # Connectivity matrix accordance
        self.tracking_connectivity_score_monitor = BatchHistoryMonitor(
            'tracking_connectivity_score_monitor', weighted=True)

        if self.add_a_tracking_validation_phase:
            new_monitors = [self.tracking_very_good_IS_monitor,
                            self.tracking_acceptable_IS_monitor,
                            self.tracking_very_far_IS_monitor,
                            self.tracking_valid_diverg_monitor,
                            self.tracking_mean_final_distance_monitor,
                            self.tracking_clipped_final_distance_monitor,
                            self.tracking_connectivity_score_monitor]
            self.monitors += new_monitors
            self.validation_monitors += new_monitors

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'add_a_tracking_validation_phase': self.add_a_tracking_validation_phase,
            'tracking_phase_frequency': self.tracking_phase_frequency,
            'tracking_phase_nb_steps_init': self.tracking_phase_nb_steps_init,
            'tracking_phase_mask_group': self.tracking_mask_group,
        })

        return p

    def _get_latest_loss_to_supervise_best(self):
        if self.use_validation:
            if self.add_a_tracking_validation_phase:
                # Choosing connectivity.
                mean_epoch_loss = \
                    self.tracking_connectivity_score_monitor.average_per_epoch[-1]
            else:
                mean_epoch_loss = self.valid_local_loss_monitor.average_per_epoch[-1]
        else:
            # Without a validation set: take the training loss.
            mean_epoch_loss = self.train_loss_monitor.average_per_epoch[-1]

        return mean_epoch_loss

    def validate_one_batch(self, data, epoch):
        # 1. Compute local loss.
        super().validate_one_batch(data, epoch)

        # 2. Compute generation losses.
        if self.add_a_tracking_validation_phase:
            if (epoch + 1) % self.tracking_phase_frequency == 0:
                logger.debug("Additional tracking-like generation validation "
                             "from batch.")
                (gen_n, mean_final_dist, mean_clipped_final_dist,
                 percent_IS_very_good, percent_IS_acceptable, percent_IS_very_far,
                 diverging_pnt, connectivity) = self.generate_from_one_batch(
                    data, compute_all_scores=True)

                self.tracking_very_good_IS_monitor.update(
                    percent_IS_very_good, weight=gen_n)
                self.tracking_acceptable_IS_monitor.update(
                    percent_IS_acceptable, weight=gen_n)
                self.tracking_very_far_IS_monitor.update(
                    percent_IS_very_far, weight=gen_n)

                self.tracking_mean_final_distance_monitor.update(
                    mean_final_dist, weight=gen_n)
                self.tracking_clipped_final_distance_monitor.update(
                    mean_clipped_final_dist, weight=gen_n)
                self.tracking_valid_diverg_monitor.update(
                    diverging_pnt, weight=gen_n)

                self.tracking_connectivity_score_monitor.update(
                    connectivity, weight=gen_n)
            elif len(self.tracking_mean_final_distance_monitor.average_per_epoch) == 0:
                logger.info("Skipping tracking-like generation validation from "
                            "batch. No values yet: adding fake initial values.")
                # Fake values at the beginning
                # Bad IS = 100%
                self.tracking_very_good_IS_monitor.update(100.0)
                self.tracking_acceptable_IS_monitor.update(100.0)
                self.tracking_very_far_IS_monitor.update(100.0)

                # Bad diverging = very far from 0. Either 100% (if diverged at
                # first point) or anything >0 if diverged further than expected
                # point.
                self.tracking_valid_diverg_monitor.update(100.0)

                # Bad mean dist = very far. ex, 100, or clipped.
                self.tracking_mean_final_distance_monitor.update(100.0)
                self.tracking_clipped_final_distance_monitor.update(
                    ACCEPTABLE_THRESHOLD)

                self.tracking_connectivity_score_monitor.update(1)
            else:
                logger.info("Skipping tracking-like generation validation from "
                            "batch. Copying previous epoch's values.")
                # Copy previous value
                for monitor in [self.tracking_very_good_IS_monitor,
                                self.tracking_acceptable_IS_monitor,
                                self.tracking_very_far_IS_monitor,
                                self.tracking_valid_diverg_monitor,
                                self.tracking_mean_final_distance_monitor,
                                self.tracking_clipped_final_distance_monitor,
                                self.tracking_connectivity_score_monitor]:
                    monitor.update(monitor.average_per_epoch[-1])

    def generate_from_one_batch(self, data, compute_all_scores=False):
        # Data interpolation has not been done yet. GPU computations are done
        # here in the main thread.
        torch.set_printoptions(precision=4)
        np.set_printoptions(precision=2)

        real_lines, ids_per_subj = data
        real_lines = [line.to(self.device, non_blocking=True, dtype=torch.float)
                      for line in real_lines]
        last_pos = torch.vstack([line[-1, :] for line in real_lines])

        # Dataloader always works on CPU. Sending to right device.
        # (model is already moved). Using only the n first points
        lines = [s[0:min(len(s), self.tracking_phase_nb_steps_init), :]
                 for s in real_lines]

        # Propagation: no backward tracking.
        previous_context = self.model.context
        self.model.set_context('tracking')
        lines = self.propagate_multiple_lines(lines, ids_per_subj)
        self.model.set_context(previous_context)

        # 1. Final distance compared to expected point.
        computed_last_pos = torch.vstack([line[-1, :] for line in lines])
        l2_loss = PairwiseDistance(p=2)
        final_dist = l2_loss(computed_last_pos, last_pos)

        if not compute_all_scores:
            return final_dist
        else:
            # 1. Also clipping final dist
            final_dist_clipped = torch.clip(final_dist, min=None,
                                            max=ACCEPTABLE_THRESHOLD)
            final_dist = torch.mean(final_dist)
            final_dist_clipped = torch.mean(final_dist_clipped)

            # 2. Connectivity scores
            connectivity_score = self._compare_connectivity(lines, ids_per_subj)

            # 3. Verify "IS ratio", i.e. percentage of streamlines ending
            # inside a predefined radius.
            invalid_ratio_severe = torch.sum(
                final_dist > VERY_CLOSE_THRESHOLD) / len(lines) * 100
            invalid_ratio_acceptable = torch.sum(
                final_dist > ACCEPTABLE_THRESHOLD) / len(lines) * 100
            invalid_ratio_loose = torch.sum(
                final_dist > VERY_FAR_THRESHOLD) / len(lines) * 100

            # 4. Verify point where streamline starts diverging.
            # 0% = error at first point --> really bad.
            # 100% = reached exactly the right point.
            # >100% = went too far (longer than expected).
            # We want a decreasing value towards 0.
            # abs(100 - score): 0 = good. 100 = bad.
            # Using 100 - x, so the score is diminishing, from 100 = perfect.
            total_point = 0
            for line, real_line in zip(lines, real_lines):
                expected_nb = len(real_line)
                diff_nb = abs(len(real_line) - len(line))
                if len(line) < expected_nb:
                    diff_nb = len(real_line) - len(line)
                    line = torch.vstack((line, line[-1, :].repeat(diff_nb, 1)))
                elif len(line) > expected_nb:
                    real_line = torch.vstack((real_line,
                                              real_line[-1, :].repeat(diff_nb, 1)))
                dist = l2_loss(line, real_line).cpu().numpy()
                point, = np.where(dist > ACCEPTABLE_THRESHOLD)
                if len(point) > 0:  # (else: score = 0. Never out of range).
                    div_point = point[0] / expected_nb * 100.0
                    total_point += abs(100 - div_point)
            diverging_point = total_point / len(lines)

            invalid_ratio_severe = invalid_ratio_severe.cpu().numpy().astype(np.float32)
            invalid_ratio_acceptable = invalid_ratio_acceptable.cpu().numpy().astype(np.float32)
            invalid_ratio_loose = invalid_ratio_loose.cpu().numpy().astype(np.float32)
            final_dist = final_dist.cpu().numpy().astype(np.float32)
            final_dist_clipped = final_dist_clipped.cpu().numpy().astype(np.float32)
            diverging_point = np.asarray(diverging_point, dtype=np.float32)
            return (len(lines), final_dist, final_dist_clipped,
                    invalid_ratio_severe, invalid_ratio_acceptable,
                    invalid_ratio_loose, diverging_point,
                    connectivity_score)

    def _compare_connectivity(self, lines, ids_per_subj):
        connectivity_matrices, volume_sizes, downsampled_sizes = \
            self.batch_loader.load_batch_connectivity_matrices(ids_per_subj)

        score = 0.0
        for i, subj in enumerate(ids_per_subj.keys()):
            real_matrix = connectivity_matrices[i]
            volume_size = volume_sizes[i]
            downsampled_size = downsampled_sizes[i]
            _lines = lines[ids_per_subj[subj]]

            batch_matrix = compute_triu_connectivity(
                _lines, volume_size, downsampled_size,
                binary=False, to_sparse_tensor=False, device=self.device)

            # Where our batch has a 1, if there was really a one: score should
            # be 0. Else, score should be 1.
            # If two streamlines in a voxel, score is 0 or 2.

            # Real matrices are saved as binary in create_hdf5.
            where_one = np.where(batch_matrix > 0)
            score += np.sum(batch_matrix[where_one] *
                            (1.0 - real_matrix[where_one]))

        # Average for batch
        score = score / len(lines)
        return score

    def propagate_multiple_lines(self, lines: List[torch.Tensor], ids_per_subj):
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        def update_memory_after_removing_lines(_, __):
            pass

        def get_dirs_at_last_pos(_lines: List[torch.Tensor], n_last_pos):
            n_last_pos = [pos[None, :] for pos in n_last_pos]
            batch_inputs = self.batch_loader.load_batch_inputs(
                n_last_pos, ids_per_subj)

            if self.model.forward_uses_streamlines:
                model_outputs = self.model(batch_inputs, n_last_pos)
            else:
                model_outputs = self.model(batch_inputs)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)
        return propagate_multiple_lines(
            lines, update_memory_after_removing_lines, get_dirs_at_last_pos,
            theta=theta, step_size=self.model.step_size,
            verify_opposite_direction=False, mask=self.tracking_mask,
            max_nbr_pts=max_nbr_pts, append_last_point=False,
            normalize_directions=True)
