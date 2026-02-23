# -*- coding: utf-8 -*-
"""
Adds a generation-validation phase: a tracking step. Metrics on the streamlines
are:

- Mean distance from expected endpoint:
    In voxel space. Same drawback as above. Also, a single bad streamline may
    contribute intensively to the score.
- Connectivity fit:
    Percentage of streamlines ending in a block of the volume indeed connected
    in the validation subject. Real connectivity matrices must be saved in the
    hdf5. Right now, volumes are simply split into blocs (the same way as in
    the hdf5, ex, to 10x10x10 volumes for a total of 1000 blocks), not based on
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
from dwi_ml.experiment_utils.memory import BYTES_IN_GB
from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.io_utils import prepare_tracking_mask
from dwi_ml.training.batch_loaders import DWIMLBatchLoaderOneInput
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.monitoring import BatchHistoryMonitor

logger = logging.getLogger('train_logger')


class DWIMLTrainerOneInputWithGVPhase(DWIMLTrainerOneInput):
    model: ModelWithDirectionGetter
    batch_loader: DWIMLBatchLoaderOneInput

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

        # -------- Checks
        if add_a_tracking_validation_phase and \
                tracking_phase_mask_group is None:
            raise NotImplementedError("Not ready to run without a tracking "
                                      "mask.")

        # -------- Monitors
        # At training time: only the one metric used for training.
        # At validation time: A lot of exploratory metrics monitors.

        # Final distance from expected point
        self.tracking_mean_final_distance_monitor = BatchHistoryMonitor(
            'tracking_mean_final_distance_monitor', weighted=True)

        # Connectivity matrix accordance
        self.tracking_connectivity_score_monitor = BatchHistoryMonitor(
            'tracking_connectivity_score_monitor', weighted=True)

        if self.add_a_tracking_validation_phase:
            new_monitors = [self.tracking_mean_final_distance_monitor,
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

    def validate_one_batch(self, targets, ids_per_subj, epoch):
        # 1. Compute the local loss as usual.
        super().validate_one_batch(targets, ids_per_subj, epoch)

        logger.debug("--> Max peak during validation (forward, before "
                     "GV phase): ")
        logger.debug(torch.cuda.max_memory_allocated() / BYTES_IN_GB)

        # 2. Compute generation losses.
        if self.add_a_tracking_validation_phase:
            if (epoch + 1) % self.tracking_phase_frequency == 0:
                logger.debug("Additional tracking-like generation validation "
                             "from batch.")
                (gen_n, mean_final_dist, connectivity) = \
                    self.gv_phase_one_batch(targets, ids_per_subj)

                self.tracking_mean_final_distance_monitor.update(
                    mean_final_dist, weight=gen_n)

                if self.compute_connectivity:
                    self.tracking_connectivity_score_monitor.update(
                        connectivity, weight=gen_n)
            elif len(self.tracking_mean_final_distance_monitor.average_per_epoch) == 0:
                logger.info("Skipping tracking-like generation validation "
                            "from batch. No values yet: adding fake initial "
                            "values.")
                # Fake values at the beginning
                # Bad mean dist = very far. ex, 100, or clipped.
                self.tracking_mean_final_distance_monitor.update(100.0)

                if self.compute_connectivity:
                    self.tracking_connectivity_score_monitor.update(1)
            else:
                logger.info("Skipping tracking-like generation validation "
                            "from batch. Copying previous epoch's values.")
                # Copy previous value
                for monitor in [self.tracking_mean_final_distance_monitor,
                                self.tracking_connectivity_score_monitor]:
                    monitor.update(monitor.average_per_epoch[-1])

    def gv_phase_one_batch(self, targets, ids_per_subj):
        """
        Use tractography to generate streamlines starting from the "true"
        seeds and first few segments. Expected results are the batch's
        validation streamlines.
        """
        # Possibly sending again to GPU even if done in the local loss
        # computation, but easier with current implementation.
        targets = [line.to(self.device, non_blocking=True,
                              dtype=torch.float)
                      for line in targets]
        last_pos = torch.vstack([line[-1, :] for line in targets])

        # Starting from the n first segments.
        # Ex: 1 segment = seed + 1 point = 2 points = s[0:2]
        lines = [s[0:min(len(s), self.tracking_phase_nb_segments_init + 1), :]
                 for s in targets]

        # Propagation: no backward tracking.
        previous_context = self.model.context
        self.model.set_context('tracking')
        lines = self.propagate_multiple_lines(lines, ids_per_subj)
        self.model.set_context(previous_context)

        # 1. Final distance compared to expected point.
        computed_last_pos = torch.vstack([line[-1, :] for line in lines])
        l2_loss = PairwiseDistance(p=2)
        final_dist = l2_loss(computed_last_pos, last_pos)
        final_dist = torch.mean(final_dist)
        final_dist = final_dist.item()

        # 2. Connectivity scores, if available (else None)
        connectivity_score = self._compare_connectivity(lines, ids_per_subj)

        return len(lines), final_dist, connectivity_score

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
                self.batch_loader.load_batch_connectivity_matrices(
                    ids_per_subj)

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
                    # Note: uses streamlines in vox space, corner origin
                    batch_matrix, _, _, _ =\
                        compute_triu_connectivity_from_labels(
                            _lines, labels, use_scilpy=False)

                if batch_matrix.shape[0] != real_matrix.shape[0]:
                    raise ValueError(
                        "You do not seem to be using the same labels ({} "
                        "labels) for the connectivity matrix as what used to "
                        "compute the reference connectivity matrices in the "
                        "hdf5 (nb rows: {})."
                        .format(batch_matrix[0].shape, real_matrix.shape[0]))

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

    def propagate_multiple_lines(self, lines: List[torch.Tensor],
                                 ids_per_subj):
        """
        Tractography propagation of 'lines'.
        Supposing the model receives the current input and current position.
        """
        assert self.model.step_size is not None, \
            "We can't propagate compressed streamlines."

        # Setting our own limits here.
        theta = 2 * np.pi  # theta = 360 degrees
        max_nbr_pts = int(200 / self.model.step_size)

        # These methods will be used during the loop on subjects
        # Based on the tracker
        def update_memory_after_removing_lines(_, __):
            pass

        def get_dirs_at_last_pos(subj_lines: List[torch.Tensor], n_last_pos):
            nonlocal subj_idx

            n_last_pos = [pos[None, :] for pos in n_last_pos]
            subj_dict = {subj_idx: slice(0, len(n_last_pos))}
            subj_inputs = self.batch_loader.load_batch_inputs(
                n_last_pos, subj_dict)

            # Supposing the model receives the current input and current
            # position.
            model_outputs = self.model(subj_inputs, n_last_pos)

            next_dirs = self.model.get_tracking_directions(
                model_outputs, algo='det', eos_stopping_thresh=0.5)
            return next_dirs

        # Running the propagation separately for each subject
        # (because they all need their own tracking mask)
        final_lines = []
        for subj_idx, line_idx in ids_per_subj.items():
            subj_id = self.batch_loader.context_subset.subjects[subj_idx]

            # Load the subject's tracking mask
            with h5py.File(self.batch_loader.dataset.hdf5_file, 'r'
                           ) as hdf_handle:
                logging.debug("Loading subj {} ({})'s tracking mask."
                              .format(subj_idx, subj_id))
                tracking_mask, _ = prepare_tracking_mask(
                    hdf_handle, self.tracking_mask_group, subj_id=subj_id,
                    mask_interp='nearest')
                tracking_mask.move_to(self.device)

            # Propagates all lines for this subject
            final_lines.extend(propagate_multiple_lines(
                lines[line_idx], update_memory_after_removing_lines,
                get_dirs_at_last_pos, theta=theta,
                step_size=self.model.step_size,
                verify_opposite_direction=False,
                mask=tracking_mask, max_nbr_pts=max_nbr_pts,
                append_last_point=False, normalize_directions=True))

        return final_lines
