# -*- coding: utf-8 -*-
"""
Adds a tracking step to verify the generation process. Metrics on the
streamlines are:

- Very good / acceptable / very far IS threshold:
    Percentage of streamlines ending inside a radius of 15 / 25 / 40 voxels of
    the expected endpoint. This metric has the drawback that streamlines
    following a correct path different from the "true" validation streamline
    contribute negatively to the metric.
- 'diverg':
    The point where the streamline becomes significantly far (i.e. > 25 voxels)
    from the "true" path. Values range between 100 (100% bad, i.e. diverging
    from the start) to 0 (0% bad; ended correclty). If the generated streamline
    is longer than the "true" one, values range between 0 (0% bad) and infinit
    (ex: 100% = went 100% too far before becoming far from the expected point.
    I.e. the generated streamline is at least twice as long as expected). Same
    drawback as above.
- Mean distance from expected endpoint:
    In voxel space. Same drawback as above. Also, a single bad streamline may
    contribute intensively to the score.
- Idem, clipped.
    Distances are clipped at 25. We consider that bad streamlines are bad, no
    matter if they end up near or far.
- Connectivity fit:
    Percentage of streamlines ending in a block of the volume indeed connected
    in the validation subject. Real connectivity matrices must be saved in the
    hdf5. Right now, volumes are simply split into blocs (the same way as in the
    hdf5, ex, to 10x10x10 volumes for a total of 1000 blocks), not based on
    anatomical ROIs. It has the advantage that it does not rely on the quality
    of segmentation. It had the drawback that a generated streamline ending
    very close to the "true" streamline, but in another block, if the
    expected endpoint is close to the border of the block, contributes
    negatively to the metric. It does not however have the drawback of other
    metrics stated before.
"""
import logging
from typing import List

import h5py
import numpy as np
import torch
from torch.nn import PairwiseDistance

from dwi_ml.data.processing.streamlines.post_processing import \
    compute_triu_connectivity_from_blocs, compute_triu_connectivity_from_labels
from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.io_utils import prepare_tracking_mask
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
                 tracking_phase_frequency: int = 1,
                 tracking_phase_nb_segments_init: int = 5,
                 tracking_phase_mask_group: str = None, *args, **kw):
        """
        Parameters
        ----------
        add_a_tracking_validation_phase: bool
            If true, the validation phase is extended with a generation (i.e.
            tracking) step: the first N points of the validation streamlines
            are kept as is, and streamlines are propagated through tractography
            until they get out of the mask, or until the EOS criteria is
            reached (if any) (threshold = 0.5).
            In current implementation, the metric defining the best model is
            the connectivity metric.
        tracking_phase_frequency: int
            There is the possibility to compute this additional step only every
            X epochs.
        tracking_phase_nb_segments_init: int
            Number of initial segments to keep in the validation step. Adding
            enough should ensure that the generated streamlines go in the same
            direction as the "true" validation streamline to generate good
            metrics. Adding 0 : only the seed point is kept.
        tracking_phase_mask_group: str
            Name of the volume group to use as tracking mask.
        """
        super().__init__(*args, **kw)

        self.add_a_tracking_validation_phase = add_a_tracking_validation_phase
        self.tracking_phase_frequency = tracking_phase_frequency
        if tracking_phase_nb_segments_init < 0:
            raise ValueError("Number of initial segments for the generation "
                             "validation phase cannot be negative.")
        self.tracking_phase_nb_segments_init = tracking_phase_nb_segments_init
        self.tracking_mask_group = tracking_phase_mask_group

        self.compute_connectivity = self.batch_loader.data_contains_connectivity

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
            'tracking_phase_nb_segments_init': self.tracking_phase_nb_segments_init,
            'tracking_phase_mask_group': self.tracking_mask_group,
        })

        return p

    def _get_latest_loss_to_supervise_best(self):
        """Using the connectivity score, if available."""
        if (self.use_validation and self.add_a_tracking_validation_phase and
                self.compute_connectivity):
            # Choosing connectivity.
            mean_epoch_loss = \
                self.tracking_connectivity_score_monitor.average_per_epoch[-1]
            return mean_epoch_loss
        else:
            return super()._get_latest_loss_to_supervise_best()

    def validate_one_batch(self, data, epoch):
        # 1. Compute the local loss as usual.
        super().validate_one_batch(data, epoch)

        # 2. Compute generation losses.
        if self.add_a_tracking_validation_phase:
            if (epoch + 1) % self.tracking_phase_frequency == 0:
                logger.debug("Additional tracking-like generation validation "
                             "from batch.")
                (gen_n, mean_final_dist, mean_clipped_final_dist,
                 percent_IS_very_good, percent_IS_acceptable, percent_IS_very_far,
                 diverging_pnt, connectivity) = self.validation_generation_one_batch(
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

    def validation_generation_one_batch(self, data, compute_all_scores=False):
        """
        Use tractography to generate streamlines starting from the "true"
        seeds and first few segments. Expected results are the batch's
        validation streamlines.
        """
        real_lines, ids_per_subj = data

        # Possibly sending again to GPU even if done in the local loss
        # computation, but easier with current implementation.
        real_lines = [line.to(self.device, non_blocking=True, dtype=torch.float)
                      for line in real_lines]
        last_pos = torch.vstack([line[-1, :] for line in real_lines])

        # Starting from the n first segments.
        # Ex: 1 segment = seed + 1 point = 2 points = s[0:2]
        lines = [s[0:min(len(s), self.tracking_phase_nb_segments_init + 1), :]
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
            # 1. (bis) Also clipping final dist
            final_dist_clipped = torch.clip(final_dist, min=None,
                                            max=ACCEPTABLE_THRESHOLD)
            final_dist_clipped = torch.mean(final_dist_clipped)

            # 2. Connectivity scores, if available (else None)
            connectivity_score = self._compare_connectivity(lines, ids_per_subj)

            # 3. "IS ratio", i.e. percentage of streamlines ending inside a
            # predefined radius.
            invalid_ratio_severe = torch.sum(
                final_dist > VERY_CLOSE_THRESHOLD) / len(lines) * 100
            invalid_ratio_acceptable = torch.sum(
                final_dist > ACCEPTABLE_THRESHOLD) / len(lines) * 100
            invalid_ratio_loose = torch.sum(
                final_dist > VERY_FAR_THRESHOLD) / len(lines) * 100
            final_dist = torch.mean(final_dist)

            # 4. Verify point where streamline starts diverging.
            # abs(100 - score): 0 = good. 100 = bad (either abs(100) -> diverged
            # at first point or abs(-100) = diverged after twice the expected
            # length.
            total_point = 0
            for line, real_line in zip(lines, real_lines):
                expected_nb = len(real_line)
                diff_nb = abs(len(real_line) - len(line))
                if len(line) < expected_nb:
                    diff_nb = len(real_line) - len(line)
                    line = torch.vstack((line, line[-1, :].repeat(diff_nb, 1)))
                elif len(line) > expected_nb:
                    real_line = torch.vstack(
                        (real_line, real_line[-1, :].repeat(diff_nb, 1)))
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
        """
        If available, computes connectivity matrices for the batch and
        compares with expected values for the subject.
        """
        if self.compute_connectivity:
            # toDo. See if it's too much to keep them all in memory. Could be
            #  done in the loop for each subject.
            (connectivity_matrices, volume_sizes,
             connectivity_nb_blocs, connectivity_labels) = \
                self.batch_loader.load_batch_connectivity_matrices(ids_per_subj)

            score = 0.0
            for i, subj in enumerate(ids_per_subj.keys()):
                real_matrix = connectivity_matrices[i]
                volume_size = volume_sizes[i]
                nb_blocs = connectivity_nb_blocs[i]
                labels = connectivity_labels[i]
                _lines = lines[ids_per_subj[subj]]

                # Move to cpu, numpy now.
                _lines = [line.cpu().numpy() for line in _lines]

                # Reference matrices are saved as binary in create_hdf5,
                # but still. Ensuring.
                real_matrix = real_matrix > 0

                # But our matrix here won't be!
                if nb_blocs is not None:
                    batch_matrix, _, _ = compute_triu_connectivity_from_blocs(
                        _lines, volume_size, nb_blocs)
                else:
                    # Note: scilpy usage not ready! Simple endpoints position
                    batch_matrix, _, _, _ =\
                        compute_triu_connectivity_from_labels(
                            _lines, labels, use_scilpy=False)

                # Where our batch has a 0: not important, maybe it was simply
                # not in this batch.
                # Where our batch has a 1, if there was really a one: score
                # should be 0.   = 1 - 1 = 1 - real
                # Else, score should be high (1).  = 1 - 0 = 1 - real
                # If two streamlines have the same connection, score is
                # either 0 or 2 for that voxel.  ==> nb * (1 - real).
                where_one = np.where(batch_matrix > 0)
                score += np.sum(batch_matrix[where_one] *
                                (1.0 - real_matrix[where_one]))

            # Average for batch
            score = score / len(lines)
        else:
            score = None
        return score

    def propagate_multiple_lines(self, lines: List[torch.Tensor], ids_per_subj):
        """
        Tractography propagation of 'lines'.
        """
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        def update_memory_after_removing_lines(_, __):
            pass

        def get_dirs_at_last_pos(_lines: List[torch.Tensor], n_last_pos):
            n_last_pos = [pos[None, :] for pos in n_last_pos]
            batch_inputs = self.batch_loader.load_batch_inputs(
                n_last_pos, ids_per_subj)

            model_outputs = self.model(batch_inputs, n_last_pos)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        # Looping on subjects because current implementation requires a single
        # tracking mask. But all the rest (get_dirs_at_last_pos, particularly)
        # work on multiple subjects because the batch loader loads input
        # according to subject id. Could refactor "propagate_multiple_line" to
        # accept multiple masks or manage it differently.
        final_lines = []
        for subj_idx, line_idx in ids_per_subj.items():
            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r') as hdf_handle:
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
                step_size=self.model.step_size, verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))

        return final_lines
