# -*- coding: utf-8 -*-
import logging

import numpy as np
from scilpy.tracking.tracker import Tracker as ScilpyTracker

from dwi_ml.data.dataset.mri_data_containers import MRIData
from dwi_ml.tracking.propagator import DWIMLAbstractPropagator, \
                                       DWIMLPropagatorOneInputAndPD
from dwi_ml.tracking.seed import SeedGeneratorGPU


class DWIMLTracker(ScilpyTracker):
    def __init__(self, propagator: DWIMLAbstractPropagator, mask: MRIData,
                 seed_generator: SeedGeneratorGPU, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, mmap_mode=None,
                 rng_seed=1234, track_forward_only=False, use_gpu=False,
                 device=None):
        """
        Parameters: See scilpy.
        ----------
        propagator: now a dwi_ml version of the propagator.
        use_gpu: added option
        """

        # Warning about the mask being an MRIData instead of DataVolume but
        # ok! Modified to be able to use as tensor more easily for torch.
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds, mmap_mode,
                         rng_seed, track_forward_only)

        # Set device
        self.use_gpu = use_gpu
        self.device = device

        # Sending model and data to device.
        self.propagator.tracking_field.move_to(device=self.device)

    def track(self):
        """
        Scilpy's Tracker.track():
            - calls _get_streamlines, or
            - deals with parallel sub-processes and calls _get_streamlines_sub,
              which call _get_streamlines for each process.

        Here adding the GPU usage. Other changes for dwi_ml will be reflected
        in _get_streamlines.
        """
        if self.use_gpu:
            self.simultanenous_tracking_on_gpu()
        else:
            # On CPU, with possibility of parallel processing.
            super().track()

    def simultanenous_tracking_on_gpu(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, self.skip)
        seeds = self.seed_generator.get_next_n_pos(
            random_generator, indices, self.skip)

        # toDo Finish preparing code to use an equivalent of
        #  _get_line_both_directions but working on many streamlines at once.
        #  See Philippe's code in vitalabai.
        raise NotImplementedError


class DWIMLTrackerOneInputAndPD(DWIMLTracker):
    def __init__(self, propagator: DWIMLPropagatorOneInputAndPD,
                 mask: MRIData, seed_generator: SeedGeneratorGPU, nbr_seeds,
                 min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                 compression_th=0.1, nbr_processes=1, save_seeds=False,
                 mmap_mode=None, rng_seed=1234, track_forward_only=False,
                 use_gpu=False):
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds,
                         mmap_mode, rng_seed, track_forward_only, use_gpu)

    def _get_line_both_directions(self, pos):
        """
        Generate a streamline from an initial position following the tracking
        parameters. Same as general tracker in scilpy, adding that the
        propagator must reverse the previous_dirs in memory when going forward.
        """
        logging.debug("---> Tracking streamline from pos {}".format(pos))

        # toDo See numpy's doc: np.random.seed:
        #  This is a convenience, legacy function.
        #  The best practice is to not reseed a BitGenerator, rather to
        #  recreate a new one. This method is here for legacy reasons.
        np.random.seed(np.uint32(hash((pos, self.rng_seed))))
        line = [pos]

        # Initialize returns true if initial directions at pos are valid.
        logging.debug("---> Reinitializing propagator")
        if self.propagator.initialize(pos, self.track_forward_only):
            logging.debug("---> Starting forward propagation")

            # Forward
            forward = self._propagate_line(True)
            if len(forward) > 0:
                forward.pop(0)
                line.extend(forward)

            # Backward
            if not self.track_forward_only:
                logging.debug("---> Starting backward propagation")

                # Special for this class. Some parameters need to be reset
                # because the models starts anew for the backward tracking.
                self.propagator.start_backward()

                backward = self._propagate_line(False)
                if len(backward) > 0:
                    line.reverse()
                    line.pop()
                    line.extend(backward)

            # Clean streamline
            if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
                return line
            return None
        elif self.min_nbr_pts == 1:
            return [pos]
        return None
