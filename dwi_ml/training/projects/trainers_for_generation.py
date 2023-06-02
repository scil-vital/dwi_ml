# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch
from torch.nn import PairwiseDistance

from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.projects.utils import prepare_tracking_mask
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.monitoring import BatchHistoryMonitor, TimeMonitor

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

    def __init__(self, add_a_tracking_validation_phase: bool = False,
                 tracking_phase_frequency: int = 5,
                 tracking_phase_nb_steps_init: int = 5,
                 tracking_phase_mask_group: str = None,
                 *args, **kw):
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
        self.tracking_valid_time_monitor = TimeMonitor()

        # Percentage of streamlines inside a radius
        self.tracking_very_good_IS_monitor = BatchHistoryMonitor(weighted=True)
        self.tracking_acceptable_IS_monitor = BatchHistoryMonitor(weighted=True)
        self.tracking_very_far_IS_monitor = BatchHistoryMonitor(weighted=True)

        # Point where the streamline start diverging from "acceptable"
        self.tracking_valid_diverg_monitor = BatchHistoryMonitor(weighted=True)

        # Final distance from expected point
        self.tracking_mean_final_distance_monitor = BatchHistoryMonitor(weighted=True)
        self.tracking_clipped_final_distance_monitor = BatchHistoryMonitor(weighted=True)

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'add_a_tracking_validation_phase': self.add_a_tracking_validation_phase,
            'tracking_phase_frequency': self.tracking_phase_frequency,
            'tracking_phase_nb_steps_init': self.tracking_phase_nb_steps_init,
            'tracking_phase_mask_group': self.tracking_mask_group
        })

        return p

    def _update_states_from_checkpoint(self, current_states):
        super()._update_states_from_checkpoint(current_states)
        self.tracking_very_good_IS_monitor.set_state(
           current_states['tracking_very_good_IS_monitor_state'])
        self.tracking_acceptable_IS_monitor.set_state(
            current_states['tracking_acceptable_IS_monitor_state'])
        self.tracking_very_far_IS_monitor.set_state(
            current_states['tracking_very_far_IS_monitor_state'])

        self.tracking_valid_diverg_monitor.set_state(
            current_states['tracking_valid_diverg_monitor_state'])

        self.tracking_mean_final_distance_monitor.set_state(
            current_states['tracking_valid_loss_monitor_state'])
        self.tracking_clipped_final_distance_monitor.set_state(
            current_states['tracking_clipped_valid_loss_monitor_state'])

    def _prepare_checkpoint_info(self) -> dict:
        checkpoint_info = super()._prepare_checkpoint_info()
        checkpoint_info['current_states'].update({
            'tracking_very_good_IS_monitor_state':
                self.tracking_very_good_IS_monitor.get_state(),
            'tracking_acceptable_IS_monitor_state':
                self.tracking_acceptable_IS_monitor.get_state(),
            'tracking_very_far_IS_monitor_state':
                self.tracking_very_far_IS_monitor.get_state(),

            'tracking_valid_diverg_monitor_state':
                self.tracking_valid_diverg_monitor.get_state(),

            'tracking_valid_loss_monitor_state':
                self.tracking_mean_final_distance_monitor.get_state(),
            'tracking_clipped_valid_loss_monitor_state':
                self.tracking_clipped_final_distance_monitor.get_state(),
        })
        return checkpoint_info

    def save_local_logs(self):
        super().save_local_logs()

        self._save_log_locally(
            self.tracking_very_good_IS_monitor.average_per_epoch,
            "tracking_validation_very_good_IS_per_epoch_{}.npy"
            .format(VERY_CLOSE_THRESHOLD))
        self._save_log_locally(
            self.tracking_acceptable_IS_monitor.average_per_epoch,
            "tracking_validation_acceptable_IS_per_epoch_{}.npy"
            .format(ACCEPTABLE_THRESHOLD))
        self._save_log_locally(
            self.tracking_very_far_IS_monitor.average_per_epoch,
            "tracking_validation_very_far_IS_per_epoch_{}.npy"
            .format(VERY_FAR_THRESHOLD))

        self._save_log_locally(
            self.tracking_valid_diverg_monitor.average_per_epoch,
            "tracking_validation_diverg_per_epoch.npy")

        self._save_log_locally(
            self.tracking_mean_final_distance_monitor.average_per_epoch,
            "tracking_validation_loss_per_epoch.npy")
        self._save_log_locally(
            self.tracking_clipped_final_distance_monitor.average_per_epoch,
            "tracking_clipped_validation_loss_per_epoch.npy")

    def validate_one_epoch(self, epoch):
        if self.add_a_tracking_validation_phase:
            self.tracking_very_good_IS_monitor.start_new_epoch()
            self.tracking_acceptable_IS_monitor.start_new_epoch()
            self.tracking_very_far_IS_monitor.start_new_epoch()
            self.tracking_valid_diverg_monitor.start_new_epoch()
            self.tracking_mean_final_distance_monitor.start_new_epoch()
            self.tracking_clipped_final_distance_monitor.start_new_epoch()
            self.tracking_valid_time_monitor.start_new_epoch()

        # This will run our modified "validate one batch" for each batch.
        super().validate_one_epoch(epoch)

        if self.add_a_tracking_validation_phase:
            self.tracking_very_good_IS_monitor.end_epoch()
            self.tracking_acceptable_IS_monitor.end_epoch()
            self.tracking_very_far_IS_monitor.end_epoch()
            self.tracking_valid_diverg_monitor.end_epoch()
            self.tracking_mean_final_distance_monitor.end_epoch()
            self.tracking_clipped_final_distance_monitor.end_epoch()
            self.tracking_valid_time_monitor.end_epoch()

            # Save info
            if self.comet_exp:
                self._update_comet_after_epoch('validation', epoch,
                                               tracking_phase=True)

    def _get_latest_loss_to_supervise_best(self):
        if self.use_validation:
            if False: # self.add_a_tracking_validation_phase:
                # Compared to super, replacing by tracking_valid loss.
                mean_epoch_loss = self.tracking_clipped_final_distance_monitor.average_per_epoch[-1]

                # Could use IS instead, or non-clipped, or diverging point.
                # Not implemented.
            else:
                mean_epoch_loss = self.valid_loss_monitor.average_per_epoch[-1]
        else:
            mean_epoch_loss = self.train_loss_monitor.average_per_epoch[-1]

        return mean_epoch_loss

    def validate_one_batch(self, data, epoch):
        mean_loss, n = super().validate_one_batch(data, epoch)

        if self.add_a_tracking_validation_phase:
            if (epoch + 1) % self.tracking_phase_frequency == 0:
                logger.info("Additional tracking-like generation validation "
                            "from batch.")
                (gen_n, mean_final_dist, mean_clipped_final_dist,
                 percent_IS_very_good, percent_IS_acceptable, percent_IS_very_far,
                 diverging_pnt) = self.generate_from_one_batch(data)

                self.tracking_very_good_IS_monitor.update(percent_IS_very_good, weight=n)
                self.tracking_acceptable_IS_monitor.update(percent_IS_acceptable, weight=n)
                self.tracking_very_far_IS_monitor.update(percent_IS_very_far, weight=n)

                self.tracking_mean_final_distance_monitor.update(mean_final_dist, weight=n)
                self.tracking_clipped_final_distance_monitor.update(mean_clipped_final_dist, weight=n)
                self.tracking_valid_diverg_monitor.update(diverging_pnt, weight=n)
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
                self.tracking_clipped_final_distance_monitor.update(ACCEPTABLE_THRESHOLD)
            else:
                logger.info("Skipping tracking-like generation validation from "
                            "batch. Copying previous epoch's values.")
                # Copy previous value
                for monitor in [self.tracking_very_good_IS_monitor,
                                self.tracking_acceptable_IS_monitor,
                                self.tracking_very_far_IS_monitor,
                                self.tracking_valid_diverg_monitor,
                                self.tracking_mean_final_distance_monitor,
                                self.tracking_clipped_final_distance_monitor]:
                    monitor.update(monitor.average_per_epoch[-1])

        return mean_loss, n

    def _update_comet_after_epoch(self, context: str, epoch: int,
                                  tracking_phase=False):
        if tracking_phase:
            torch.set_printoptions(precision=4)
            np.set_printoptions(precision=4)

            final_dist = self.tracking_mean_final_distance_monitor.average_per_epoch[-1]
            clipped = self.tracking_clipped_final_distance_monitor.average_per_epoch[-1]
            logger.info("   Mean final distance for this epoch: {}\n"
                        "       (Clipped at {}: {})"
                        .format(final_dist, ACCEPTABLE_THRESHOLD, clipped))

            percent_IS_good = self.tracking_very_good_IS_monitor.average_per_epoch[-1]
            percent_IS_ok = self.tracking_acceptable_IS_monitor.average_per_epoch[-1]
            percent_IS_bad = self.tracking_very_far_IS_monitor.average_per_epoch[-1]

            logger.info("Mean simili-IS ratio for this epoch:\n"
                        "                          Threshold {}: {}\n"
                        "                          Threshold {}: {}\n"
                        "                          Threshold {}: {}"
                        .format(VERY_CLOSE_THRESHOLD, percent_IS_good,
                                ACCEPTABLE_THRESHOLD, percent_IS_ok,
                                VERY_FAR_THRESHOLD, percent_IS_bad))

            diverg = self.tracking_valid_diverg_monitor.average_per_epoch[-1]
            logger.info("Mean diverging point for this epoch: {}\n"
                        " (percentage of streamline where distance becomes >{}, "
                        "or percentage above 100% for streamlines longer than "
                        "expected)".format(diverg, ACCEPTABLE_THRESHOLD))

            if self.comet_exp:
                comet_context = self.comet_exp.validate
                with comet_context():
                    self.comet_exp.log_metric(
                        "Mean final distance", final_dist, step=epoch)
                    self.comet_exp.log_metric(
                        "Mean final distance (clipped {})"
                        .format(ACCEPTABLE_THRESHOLD), clipped, step=epoch)
                    self.comet_exp.log_metric(
                        "IS ratio at dist {}".format(VERY_CLOSE_THRESHOLD),
                        percent_IS_good, step=epoch)
                    self.comet_exp.log_metric(
                        "IS ratio at dist {}".format(ACCEPTABLE_THRESHOLD),
                        percent_IS_ok, step=epoch)
                    self.comet_exp.log_metric(
                        "IS ratio at dist {}".format(VERY_FAR_THRESHOLD),
                        percent_IS_bad, step=epoch)
                    self.comet_exp.log_metric(
                        "Diverging point", diverg, step=epoch)

        super()._update_comet_after_epoch(context, epoch)

    def generate_from_one_batch(self, data):
        # Data interpolation has not been done yet. GPU computations are done
        # here in the main thread.
        torch.set_printoptions(precision=4)
        np.set_printoptions(precision=2)

        real_lines, ids_per_subj = data
        real_lines = [line.to(self.device, non_blocking=True, dtype=torch.float)
                      for line in real_lines]
        last_pos = torch.vstack([line[-1, :] for line in real_lines])
        mean_length = np.mean([len(s) for s in real_lines])

        # Dataloader always works on CPU. Sending to right device.
        # (model is already moved). Using only the n first points
        lines = [s[0:min(len(s), self.tracking_phase_nb_steps_init), :]
                 for s in real_lines]
        self.model.set_context('tracking')
        lines = self.propagate_multiple_lines(lines, ids_per_subj)
        self.model.set_context('validation')

        compute_mean_length = np.mean([len(s) for s in lines])
        logger.info("-> Average streamline length (nb pts) in this batch: {} \n"
                    "                   Average recovered streamline length: {}"
                    .format(mean_length.astype(np.float64),
                            compute_mean_length.astype(np.float64)))

        # 1. Final distance compared to expected point.
        computed_last_pos = torch.vstack([line[-1, :] for line in lines])
        l2_loss = PairwiseDistance(p=2)
        final_dist = l2_loss(computed_last_pos, last_pos)

        # Verify "IS ratio", i.e. percentage of streamlines ending inside a
        # predefined radius.
        IS_ratio_good = torch.sum(final_dist > VERY_CLOSE_THRESHOLD) / len(lines) * 100
        IS_ratio_ok = torch.sum(final_dist > ACCEPTABLE_THRESHOLD) / len(lines) * 100
        IS_ratio_bad = torch.sum(final_dist > VERY_FAR_THRESHOLD) / len(lines) * 100

        final_dist_clipped = torch.clip(final_dist, min=None,
                                        max=ACCEPTABLE_THRESHOLD)
        final_dist = torch.mean(final_dist)
        final_dist_clipped = torch.mean(final_dist_clipped)

        # Verify point where streamline starts diverging.
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
            dist = l2_loss(line, real_line).detach().cpu().numpy()
            point, = np.where(dist > ACCEPTABLE_THRESHOLD)
            if len(point) > 0:  # (else: score = 0. Never out of range).
                div_point = point[0] / expected_nb * 100.0
                total_point += abs(100 - div_point)
        diverging_point = total_point / len(lines)

        IS_ratio_good = IS_ratio_good.cpu().numpy().astype(np.float32)
        IS_ratio_ok = IS_ratio_ok.cpu().numpy().astype(np.float32)
        IS_ratio_bad = IS_ratio_bad.cpu().numpy().astype(np.float32)
        final_dist = final_dist.cpu().numpy().astype(np.float32)
        final_dist_clipped = final_dist_clipped.cpu().numpy().astype(np.float32)
        diverging_point = np.asarray(diverging_point, dtype=np.float32)
        return (len(lines), final_dist, final_dist_clipped,
                IS_ratio_good, IS_ratio_ok, IS_ratio_bad, diverging_point)

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
