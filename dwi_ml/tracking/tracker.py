# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os

import numpy as np
import torch
from scilpy.image.datasets import DataVolume
from scilpy.tracking.tracker import Tracker as ScilpyTracker

from dwi_ml.tracking.propagator import DWIMLPropagator
from dwi_ml.tracking.seed import DWIMLSeedGenerator

logger = logging.getLogger('tracker_logger')


class DWIMLTracker(ScilpyTracker):
    def __init__(self, propagator: DWIMLPropagator, mask: DataVolume,
                 seed_generator: DWIMLSeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, rng_seed=1234,
                 track_forward_only=False, simultanenous_tracking: int = 1,
                 use_gpu: bool = False, log_level=logging.WARNING):
        """
        Parameters: See scilpy.
        ----------
        propagator: now a dwi_ml version of the propagator.
        simultaneous_tracking: bool
            If true, track multiple lines at the same time. Intended for GPU.
        use_gpu: bool
            New option.
        hdf_file:
            Necessary to open a new hdf handle a at process.
        """

        # Warning about the mask being an MRIData instead of DataVolume but
        # ok! Modified to be able to use as tensor more easily for torch.
        mmap_mode = None  # Not used here, we deal with hdf5.
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds, mmap_mode,
                         rng_seed, track_forward_only)

        if nbr_processes > 2 and not propagator.dataset.is_lazy:
            raise ValueError("Multiprocessing only works with lazy data.")

        # Set device
        self.simultanenous_tracking = simultanenous_tracking
        self.use_gpy = use_gpu

        if use_gpu:
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
                logging.info("We will be using GPU!")
            else:
                raise ValueError("You chose GPU (cuda) device but it is not "
                                 "available!")
        else:
            self.device = torch.device('cpu')

        # Sending model and data to device.
        self.propagator.move_to(self.device)

        logger.setLevel(log_level)

        # Increase the printing frequency as compared to super
        self.printing_frequency = 5

    def track(self):
        """
        Scilpy's Tracker.track():
            - calls _get_streamlines, or
            - deals with parallel sub-processes and calls _get_streamlines_sub,
              which call _get_streamlines for each process.

        Here adding the GPU usage. Other changes for dwi_ml will be reflected
        in _get_streamlines.
        """
        if self.simultanenous_tracking > 1:
            self.simultanenous_tracking_on_gpu()
        else:
            # On CPU, with possibility of parallel processing.
            return super().track()

    def _prepare_multiprocessing_pool(self, tmpdir=None):
        """
        Prepare multiprocessing pool. We do not need to deal with data
        management, contrary to scilpy.
        """
        # Clearing data from memory
        self.propagator.reset_data(reload_data=False)

        pool = multiprocessing.Pool(self.nbr_processes)
        return pool

    def _reload_data_for_new_process(self, init_args):
        """Nothing to do here. hdf5 deals well with multi-processes."""

        logger.info("Preparing for process id {}".format(os.getpid()))
        self.propagator.reset_data(reload_data=True)

    def simultanenous_tracking_on_gpu(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        assert torch.cuda.is_available()
        assert self.device.type == 'cuda'
        self.propagator.move_to(self.device)

        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, self.skip)

        seed_count = 0
        lines = []
        while seed_count < self.nbr_seeds:
            nb_next_seeds = self.simultanenous_tracking
            if seed_count + nb_next_seeds > self.nbr_seeds:
                nb_next_seeds = self.nbr_seeds - seed_count

            logger.info("Multiple GPU tracking: Tracking the next {} "
                        "streamlines.".format(nb_next_seeds))

            next_seeds = np.asarray(
                range(seed_count, seed_count+nb_next_seeds))

            # Using DWIMLSeedGenerator's class
            seeds = self.seed_generator.get_next_n_pos(
                random_generator, indices, next_seeds)

            lines.append(self._get_multiple_lines_both_directions(seeds))

            logger.info("Done.")
            seed_count += nb_next_seeds

    def _get_multiple_lines_both_directions(self, n_seeds):
        """
        Equivalent of super's() _get_line_both_directions() for multiple lines
        tracking at the same time (meant to be used on GPU).

        Params
        ------
        seeding_pos : tuple
            3D position, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """
        lines = [[np.asarray(seeding_pos)] for seeding_pos in n_seeds]

        logger.info("Multiple GPU tracking: Starting forward propgagation")
        tracking_info = self.propagator.prepare_forward(n_seeds)
        lines = self._propagate_multiple_lines(lines, tracking_info)

        if not self.track_forward_only:
            logger.info("Multiple GPU tracking: Starting backward "
                        "propgagation")

            lines = [line.reverse() if len(line) > 1 else line
                     for line in lines]

            tracking_info = self.propagator.prepare_backward(lines,
                                                             tracking_info)
            lines = self._propagate_line(lines, tracking_info)

        # Clean streamlines
        clean_lines = []
        for line in lines:
            if self.min_nbr_pts <= len(line) <= self.max_nbr_pts:
                clean_lines.append(line)
        return clean_lines

    def _propagate_multiple_lines(self, lines, tracking_info):
        """
        Equivalent of super's() _propagate_lines() for for multiple tracking at
        the same time (meant to be used on GPU).
        """
        invalid_direction_counts = np.zeros(len(lines))
        nb_streamlines = len(lines)
        final_lines = []  # Will get the final lines when they are done.
        final_tracking_info = []

        nb_points = 0
        all_lines_completed = False
        # lines will only contain the remaining lines.
        while nb_points < self.max_nbr_pts and not all_lines_completed:
            nb_points += 1

            n_new_pos, new_tracking_info, are_directions_valid = \
                self.propagator.propagate(lines, tracking_info,
                                          multiple_lines=True)
            invalid_direction_counts[~are_directions_valid] += 1

            # Verifying and appending
            all_lines_completed = True
            old_nb = len(lines)
            for i in reversed(range(len(lines))):
                propagation_can_continue = self._verify_stopping_criteria(
                    invalid_direction_counts[i], n_new_pos[i])

                if propagation_can_continue:
                    all_lines_completed = False
                    lines[i].append(n_new_pos[i])
                else:
                    final_lines.append(lines[i])
                    lines.pop(i)
                    final_tracking_info.append(new_tracking_info[i])
                    new_tracking_info.pop(i)
                    invalid_direction_counts = np.delete(
                        invalid_direction_counts, i)

            logging.debug("Continuing propagation for the remaining {}/{} "
                          "streamlines.".format(len(lines), old_nb))
            tracking_info = new_tracking_info

        assert len(lines) == 0
        assert len(final_lines) == nb_streamlines

        # Possible last step.
        # Looping. It should not be heavy.
        for i in range(len(final_lines)):
            final_pos = self.propagator.finalize_streamline(
                final_lines[i][-1], final_tracking_info[i])
            if (final_pos is not None and
                not np.array_equal(final_pos, final_lines[i][-1]) and
                    self.mask.is_voxmm_in_bound(*final_pos,
                                                origin=self.origin)):
                final_lines[i].append(final_pos)
        return final_lines
