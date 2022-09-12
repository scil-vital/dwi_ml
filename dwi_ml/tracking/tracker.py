# -*- coding: utf-8 -*-
import logging
import multiprocessing
import os

import numpy as np
import torch
from dipy.io.stateful_tractogram import Space, Origin
from scilpy.image.datasets import DataVolume
from scilpy.tracking.tracker import Tracker as ScilpyTracker
from scilpy.tracking.seed import SeedGenerator

from dwi_ml.tracking.propagator import DWIMLPropagator

logger = logging.getLogger('tracker_logger')


class DWIMLTracker(ScilpyTracker):
    # Removing a few warning by explicitly telling python that the
    # propagator type is not as in super. Supported in python > 3.5
    propagator: DWIMLPropagator

    def __init__(self, propagator: DWIMLPropagator, mask: DataVolume,
                 seed_generator: SeedGenerator, nbr_seeds, min_nbr_pts,
                 max_nbr_pts, max_invalid_dirs, compression_th=0.1,
                 nbr_processes=1, save_seeds=False, rng_seed=1234,
                 track_forward_only=False, simultanenous_tracking: int = 1,
                 use_gpu: bool = False, log_level=logging.WARNING):
        """
        Parameters: See scilpy.
        ----------
        propagator: DWIMLPropagator
            Now a dwi_ml version of the propagator.
        simultaneous_tracking: bool
            If true, track multiple lines at the same time. Intended for GPU.
        use_gpu: bool
            New option.
        hdf_file:
            Necessary to open a new hdf handle a at process.
        """

        # Warning about the mask being an MRIData instead of DataVolume but
        # ok! Modified to be able to use as tensor more easily for torch.
        mmap_mode = ''  # Not used here, we deal with hdf5.
        super().__init__(propagator, mask, seed_generator, nbr_seeds,
                         min_nbr_pts, max_nbr_pts, max_invalid_dirs,
                         compression_th, nbr_processes, save_seeds, mmap_mode,
                         rng_seed, track_forward_only)

        if nbr_processes > 2:
            if not propagator.dataset.is_lazy:
                raise ValueError("Multiprocessing only works with lazy data.")
            if use_gpu:
                raise ValueError("You cannot use both multi-processes and "
                                 "gpu.")

        # Set device
        self.simultanenous_tracking = simultanenous_tracking
        self.use_gpu = use_gpu

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
        self.printing_frequency = 1

        # NOTE: TRAINER USES STREAMLINES COORDINATES IN VOXEL SPACE, TO CORNER.
        assert self.space == Space.VOX
        assert self.origin == Origin('corner')

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
            return self.simultanenous_tracking_on_gpu()
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
        seeds = []
        while seed_count < self.nbr_seeds:
            nb_next_seeds = self.simultanenous_tracking
            if seed_count + nb_next_seeds > self.nbr_seeds:
                nb_next_seeds = self.nbr_seeds - seed_count

            logger.info("*** Multiple GPU tracking: Tracking the next {} "
                        "streamlines.".format(nb_next_seeds))

            next_seeds = np.asarray(
                range(seed_count, seed_count + nb_next_seeds))

            # Using DWIMLSeedGenerator's class
            n_seeds = self.seed_generator.get_next_n_pos(
                random_generator, indices, next_seeds)

            tmp_lines, tmp_seeds = self._get_multiple_lines_both_directions(
                n_seeds)
            lines.extend(tmp_lines)
            seeds.extend(tmp_seeds)

            seed_count += nb_next_seeds
        return lines, seeds

    def _get_multiple_lines_both_directions(self, n_seeds):
        """
        Equivalent of super's() _get_line_both_directions() for multiple lines
        tracking at the same time (meant to be used on GPU).

        Params
        ------
        n_seeds : list[tuple]
            3D positions, the seed position.

        Returns
        -------
        line: list of 3D positions
            The generated streamline for seeding_pos.
        """
        lines = [[list(seeding_pos)] for seeding_pos in n_seeds]

        logger.info("Multiple GPU tracking: Starting forward propagation for "
                    "the next {} streamlines.".format(len(n_seeds)))
        tracking_info = self.propagator.prepare_forward(n_seeds)
        lines, order1 = self._propagate_multiple_lines(lines, tracking_info)

        if not self.track_forward_only:

            # Reversing in place
            for i in range(len(lines)):
                if len(lines[i]) > 1:
                    lines[i].reverse()

            # We could loop to prepare reverse. Basic case not too heavy.
            # However, in some cases (ex, Learn2track with memory), model
            # needs to be re-run before starting back-propagation. We let
            # the propagator deal with the looping.
            tracking_info = self.propagator.prepare_backward(
                lines, tracking_info, multiple_lines=True)

            logger.info("Multiple GPU tracking: Starting backward "
                        "propagation.")
            lines, order2 = self._propagate_multiple_lines(
                lines, tracking_info)

            if self.save_seeds:
                final_order = [order1[idx] for idx in order2]
        elif self.save_seeds:
            final_order = order1

        # Clean streamlines
        clean_lines = []
        clean_seeds = []
        for i in range(len(lines)):
            if self.min_nbr_pts <= len(lines[i]) <= self.max_nbr_pts:
                clean_lines.append(np.asarray(lines[i]))
                if self.save_seeds:
                    clean_seeds.append(n_seeds[final_order[i]])

        return clean_lines, clean_seeds

    def _propagate_multiple_lines(self, lines, tracking_info):
        """
        Equivalent of super's() _propagate_lines() for for multiple tracking at
        the same time (meant to be used on GPU).
        """
        nb_streamlines = len(lines)
        invalid_direction_counts = np.zeros(nb_streamlines)
        final_lines = []  # Will get the final lines when they are done.
        final_tracking_info = []
        # Note! We modify the order of streamlines in final_lines!
        # We need to remember the order to save the seeds.
        initial_order = list(range(nb_streamlines))
        final_lines_order = []

        all_lines_completed = False
        # lines will only contain the remaining lines.
        while not all_lines_completed:
            n_new_pos, tracking_info, are_directions_valid = \
                self.propagator.propagate_multiple_lines(lines, tracking_info)
            invalid_direction_counts[~are_directions_valid] += 1

            # ToDo. Check what is faster. Removing finished streamlines or
            #  continue with all streamlines but remember where they ended.
            # Verifying and appending
            all_lines_completed = True
            lines_that_continue = []
            lines_that_stop = []
            for i in range(len(lines)):
                # In non-simultaneous, nbr pts is verified in the loop.
                # Here, during forward, we can do that, but during backward,
                # all streamlines have different lengths.
                propagation_can_continue = (self._verify_stopping_criteria(
                    invalid_direction_counts[i], n_new_pos[i]) and
                    len(lines[i]) <= self.max_nbr_pts)

                if propagation_can_continue:
                    all_lines_completed = False
                    lines[i].append(n_new_pos[i])
                    lines_that_continue.append(i)
                else:
                    lines_that_stop.append(i)

            # Note. Indexing would be simpler with lines as an array but now
            # propagator is coded to deal with a list.
            final_lines.extend([lines[i] for i in lines_that_stop])
            lines = [lines[i] for i in lines_that_continue]

            final_lines_order.extend(
                [initial_order[i] for i in lines_that_stop])
            initial_order = \
                [initial_order[i] for i in lines_that_continue]

            final_tracking_info.extend(
                [tracking_info[i] for i in lines_that_stop])
            tracking_info = [tracking_info[i] for i in lines_that_continue]

            invalid_direction_counts = np.asanyarray(
                [invalid_direction_counts[i] for i in lines_that_continue])

            self.propagator.multiple_lines_update(lines_that_continue)

        assert len(lines) == 0
        assert len(initial_order) == 0
        assert len(final_lines) == nb_streamlines
        assert np.array_equal(np.arange(nb_streamlines),
                              np.unique(final_lines_order))

        # Possible last step.
        # Looping. It should not be heavy.
        for i in range(len(final_lines)):
            final_pos = self.propagator.finalize_streamline(
                final_lines[i][-1], final_tracking_info[i])
            if (final_pos is not None and
                not np.array_equal(final_pos, final_lines[i][-1]) and
                    self.mask.is_coordinate_in_bound(
                        *final_pos, space=self.space, origin=self.origin)):
                final_lines[i].append(final_pos)
        return final_lines, final_lines_order
