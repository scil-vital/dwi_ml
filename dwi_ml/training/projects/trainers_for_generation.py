# -*- coding: utf-8 -*-
import logging
from typing import List

import h5py
import numpy as np
import torch
from torch.nn import PairwiseDistance
from tqdm import tqdm

from dwi_ml.experiment_utils.tqdm_logging import tqdm_logging_redirect
from dwi_ml.models.main_models import ModelWithDirectionGetter
from dwi_ml.tracking.propagation import propagate_multiple_lines
from dwi_ml.tracking.projects.utils import prepare_tracking_mask
from dwi_ml.training.trainers import DWIMLTrainerOneInput
from dwi_ml.training.utils.monitoring import BatchHistoryMonitor, TimeMonitor

logger = logging.getLogger('train_logger')

# Emma tests in ISMRM: a box of 30x30x30 mm sounds good.
# So half of it, max distance = sqrt( 3 * 15^2) =
IS_THRESHOLD = 25.98


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
        self.tracking_valid_time_monitor = TimeMonitor()
        self.tracking_valid_IS_monitor = BatchHistoryMonitor(weighted=True)
        self.tracking_valid_loss_monitor = BatchHistoryMonitor(weighted=True)
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
        self.tracking_valid_loss_monitor.set_state(
            current_states['tracking_valid_loss_monitor_state'])
        self.tracking_valid_IS_monitor.set_state(
           current_states['tracking_valid_IS_monitor_state'])

    def _prepare_checkpoint_info(self) -> dict:
        checkpoint_info = super()._prepare_checkpoint_info()
        checkpoint_info['current_states'].update({
            'tracking_valid_loss_monitor_state':
                self.tracking_valid_loss_monitor.get_state(),
            'tracking_valid_IS_monitor_state':
                self.tracking_valid_IS_monitor.get_state(),
        })
        return checkpoint_info

    def save_local_logs(self):
        super().save_local_logs()
        self._save_log_locally(
            self.tracking_valid_loss_monitor.average_per_epoch,
            "tracking_validation_loss_per_epoch.npy")
        self._save_log_locally(
            self.tracking_valid_IS_monitor.average_per_epoch,
            "tracking_validation_IS_per_epoch.npy")

    def validate_one_epoch(self, epoch):
        if self.add_a_tracking_validation_phase:
            self.tracking_valid_loss_monitor.start_new_epoch()
            self.tracking_valid_IS_monitor.start_new_epoch()
            self.tracking_valid_time_monitor.start_new_epoch()

        super().validate_one_epoch(epoch)

        if self.add_a_tracking_validation_phase:
            self.tracking_valid_loss_monitor.end_epoch()
            self.tracking_valid_IS_monitor.end_epoch()
            self.tracking_valid_time_monitor.end_epoch()

            # Save info
            if self.comet_exp:
                self._update_comet_after_epoch(self.comet_exp.validate, epoch,
                                               tracking_phase=True)

    def _get_latest_loss_to_supervise_best(self):
        if self.use_validation:
            if self.add_a_tracking_validation_phase:
                # Compared to super, replacing by tracking_valid loss.
                mean_epoch_loss = self.tracking_valid_loss_monitor.average_per_epoch[-1]

                # Could use IS instead. Not implemented.
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
                gen_mean_loss, gen_n, percent_inv = \
                    self.generate_from_one_batch(data)
                gen_mean_loss = gen_mean_loss.cpu().item()
                self.tracking_valid_loss_monitor.update(gen_mean_loss, weight=n)
                self.tracking_valid_IS_monitor.update(percent_inv, weight=n)
            else:
                self.tracking_valid_loss_monitor.update(
                    self.tracking_valid_loss_monitor.average_per_epoch[-1])
                self.tracking_valid_IS_monitor.update(
                    self.tracking_valid_IS_monitor.average_per_epoch[-1])

        return mean_loss, n

    def _update_comet_after_epoch(self, context: str, epoch: int,
                                  tracking_phase=False):
        if tracking_phase:
            loss = self.tracking_valid_loss_monitor.average_per_epoch[-1]
            logger.info("   Mean tracking loss for this epoch: {}".format(loss))

            percent_inv = self.tracking_valid_IS_monitor.average_per_epoch[-1]
            logger.info("   Mean simili-IS ratio for this epoch: {}"
                        " (threshold {})".format(percent_inv, IS_THRESHOLD))

            if self.comet_exp:
                comet_context = self.comet_exp.validate
                with comet_context():
                    self.comet_exp.log_metric(
                        "generation_loss_per_epoch", loss, step=epoch)
                    self.comet_exp.log_metric(
                        "generation_IS_ratio_per_epoch", percent_inv, step=epoch)

        else:
            super()._update_comet_after_epoch(context, epoch)

    def generate_from_one_batch(self, data):
        # Data interpolation has not been done yet. GPU computations are done
        # here in the main thread.
        torch.set_printoptions(precision=4)
        np.set_printoptions(precision=4)

        lines, ids_per_subj = data
        lines = [line.to(self.device, non_blocking=True, dtype=torch.float)
                 for line in lines]
        last_pos = torch.vstack([line[-1, :] for line in lines])
        mean_length = np.mean([len(s) for s in lines])

        # Dataloader always works on CPU. Sending to right device.
        # (model is already moved). Using only the n first points
        lines = [s[0:min(len(s), self.tracking_phase_nb_steps_init), :]
                 for s in lines]
        lines = self.propagate_multiple_lines(lines, ids_per_subj)

        # Verify "loss", i.e. the differences in coordinates
        computed_last_pos = torch.vstack([line[-1, :] for line in lines])
        compute_mean_length = np.mean([len(s) for s in lines])

        logging.debug("   Average streamline length (nb pts) in this batch: {} \n"
                      "                Average recovered streamline length: {}"
                      .format(mean_length, compute_mean_length))
        l2_loss = PairwiseDistance(p=2)
        loss = l2_loss(computed_last_pos, last_pos)

        logging.info("    Best / Worst loss: {} / {}"
                     .format(torch.max(loss), torch.min(loss)))

        IS_ratio = torch.sum(loss > IS_THRESHOLD).cpu() / len(lines) * 100

        return torch.mean(loss), len(lines), IS_ratio

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
