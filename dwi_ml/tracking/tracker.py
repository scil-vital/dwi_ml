# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback
from typing import List

import numpy as np
import torch
from dipy.io.stateful_tractogram import Space, Origin
from dipy.tracking.streamlinespeed import compress_streamlines
from scilpy.image.datasets import DataVolume
from scilpy.tracking.seed import SeedGenerator

from dwi_ml.tracking.propagator import DWIMLPropagator

logger = logging.getLogger('tracker_logger')


class DWIMLTracker:
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
            Your propagator.
        simultaneous_tracking: bool
            If true, track multiple lines at the same time. Intended for GPU.
        use_gpu: bool
            New option.
        hdf_file:
            Necessary to open a new hdf handle a at process.
        """
        self.propagator = propagator
        self.mask = mask
        self.seed_generator = seed_generator
        self.nbr_seeds = nbr_seeds
        self.min_nbr_pts = min_nbr_pts
        self.max_nbr_pts = max_nbr_pts
        self.max_invalid_dirs = max_invalid_dirs
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.mmap_mode = None
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.skip = 0  # Not supported yet.
        self.printing_frequency = 1

        # Space and origin
        self.origin = self.propagator.origin
        self.space = self.propagator.space
        assert self.space == Space.VOX
        assert self.origin == Origin('corner')
        if (seed_generator.origin != propagator.origin or
                seed_generator.space != propagator.space):
            raise ValueError("Seed generator and propagator must work with "
                             "the same space and origin!")

        # Nb points
        if self.min_nbr_pts <= 0:
            logging.warning("Minimum number of points cannot be 0. Changed to "
                            "1.")
            self.min_nbr_pts = 1

        # Either GPU or multi-processes
        self.nbr_processes = self._set_nbr_processes(nbr_processes)
        if nbr_processes > 2:
            if not propagator.dataset.is_lazy:
                raise ValueError("Multiprocessing only works with lazy data.")
            if use_gpu:
                raise ValueError("You cannot use both multi-processes and "
                                 "gpu.")

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
        self.propagator.move_to(self.device)  # Sends model and data to device

        logger.setLevel(log_level)

    def _set_nbr_processes(self, nbr_processes):
        """
        Copied from scilpy's tracker.

        If user did not define the number of processes, define it automatically
        (or set to 1 -- no multiprocessing -- if we can't).
        """
        if nbr_processes <= 0:
            try:
                nbr_processes = multiprocessing.cpu_count()
            except NotImplementedError:
                logging.warning("Cannot determine number of cpus: "
                                "nbr_processes set to 1.")
                nbr_processes = 1

        if nbr_processes > self.nbr_seeds:
            nbr_processes = self.nbr_seeds
            logging.debug("Setting number of processes to {} since there were "
                          "less seeds than processes.".format(nbr_processes))
        return nbr_processes

    def track(self):
        """
        Scilpy's Tracker.track():
            - calls _cpu_tracking, or
            - deals with parallel sub-processes and calls _cpu_tracking_sub,
              which call _cpu_tracking for each process.

        Here adding the GPU usage. Other changes for dwi_ml will be reflected
        in _cpu_tracking.
        """
        if self.simultanenous_tracking > 1:
            return self._gpu_simultanenous_tracking()
        else:
            # On CPU, with possibility of parallel processing.
            # Copied from scilpy's tracker.
            if self.nbr_processes < 2:
                chunk_id = 1
                lines, seeds = self._cpu_tracking(chunk_id)
            else:
                # Each process will use get_streamlines_at_seeds
                chunk_ids = np.arange(self.nbr_processes)

                pool = self._cpu_prepare_multiprocessing_pool()

                lines_per_process, seeds_per_process = zip(*pool.map(
                    self._cpu_tracking_sub, chunk_ids))
                pool.close()
                # Make sure all worker processes have exited before leaving
                # context manager.
                pool.join()
                lines = [line for line in itertools.chain(*lines_per_process)]
                seeds = [seed for seed in itertools.chain(*seeds_per_process)]

            return lines, seeds

    def _cpu_prepare_multiprocessing_pool(self):
        """
        Prepare multiprocessing pool. We do not need to deal with data
        management, contrary to scilpy.
        """
        # Clearing data from memory
        self.propagator.reset_data()

        pool = multiprocessing.Pool(self.nbr_processes)
        return pool

    def _cpu_reload_data_for_new_process(self):
        """Nothing to do here. hdf5 deals well with multi-processes."""

        logger.info("Preparing for process id {}".format(os.getpid()))
        self.propagator.reset_data()

    def _cpu_tracking_sub(self, chunk_id):
        """
        multiprocessing.pool.map input function. Calls the main tracking
        method (_cpu_tracking) with correct initialization arguments
        (taken from the global variable multiprocess_init_args).

        Parameters
        ----------
        chunk_id: int
            This processes's id.

        Return
        -------
        lines: list
            List of list of 3D positions (streamlines).
        """
        self._cpu_reload_data_for_new_process()
        try:
            streamlines, seeds = self._cpu_tracking(chunk_id)
            return streamlines, seeds
        except Exception as e:
            logging.error("Operation _cpu_tracking_sub() failed.")
            traceback.print_exception(*sys.exc_info(), file=sys.stderr)
            raise e

    def _cpu_tracking(self, chunk_id):
        """
        Tracks the n streamlines associates with current process (identified by
        chunk_id). The number n is the total number of seeds / the number of
        processes. If asked by user, may compress the streamlines and save the
        seeds.

        Parameters
        ----------
        chunk_id: int
            This process ID.

        Returns
        -------
        streamlines: list
            The successful streamlines.
        seeds: list
            The list of seeds for each streamline, if self.save_seeds. Else, an
            empty list.
        """
        streamlines = []
        seeds = []

        # Initialize the random number generator to cover multiprocessing,
        # skip, which voxel to seed and the subvoxel random position
        chunk_size = int(self.nbr_seeds / self.nbr_processes)
        first_seed_of_chunk = chunk_id * chunk_size + self.skip
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, first_seed_of_chunk)
        if chunk_id == self.nbr_processes - 1:
            chunk_size += self.nbr_seeds % self.nbr_processes

        # Getting streamlines
        for s in range(chunk_size):
            if s % self.printing_frequency == 0:
                logging.info("Process {} (id {}): {} / {}"
                             .format(chunk_id, os.getpid(), s, chunk_size))

            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Forward and backward tracking
            line = self._get_multiple_lines_both_directions([seed])[0]

            if line is not None:
                streamline = np.array(line, dtype='float32')

                if self.compression_th and self.compression_th > 0:
                    # Compressing. Threshold is in mm. Verifying space.
                    if self.space == Space.VOX:
                        # Equivalent of sft.to_voxmm:
                        streamline *= self.seed_generator.voxres
                        compress_streamlines(streamline, self.compression_th)
                        # Equivalent of sft.to_vox:
                        streamline /= self.seed_generator.voxres
                    else:
                        compress_streamlines(streamline, self.compression_th)

                streamlines.append(streamline)

                if self.save_seeds:
                    seeds.append(np.asarray(seed, dtype='float32'))

        return streamlines, seeds
        
    def _gpu_simultanenous_tracking(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        assert torch.cuda.is_available()
        assert self.device.type == 'cuda'

        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, self.skip)

        seed_count = 0
        lines = []
        seeds = []
        while seed_count < self.nbr_seeds:
            nb_next_seeds = self.simultanenous_tracking
            if seed_count + nb_next_seeds > self.nbr_seeds:
                nb_next_seeds = self.nbr_seeds - seed_count

            next_seeds = np.asarray(
                range(seed_count, seed_count + nb_next_seeds))

            n_seeds = self.seed_generator.get_next_n_pos(
                random_generator, indices, next_seeds)

            tmp_lines, tmp_seeds = self._get_multiple_lines_both_directions(
                n_seeds)
            lines.extend([line.tolist() for line in tmp_lines])
            seeds.extend([s.tolist() for s in tmp_seeds])

            seed_count += nb_next_seeds

        return lines, seeds

    def _get_multiple_lines_both_directions(self, n_seeds: List[np.ndarray]):
        """
        Params
        ------
        n_seeds : List[np.ndarray]
            3D positions, the seed positions.

        Returns
        -------
        clean_lines: List[np.ndarray]
            The generated streamline for each seeding_pos.
        """
        torch.cuda.empty_cache()

        # List of list. Sending to torch tensors.
        n_seeds = [torch.as_tensor(s, device=self.device, dtype=torch.float)
                   for s in n_seeds]
        lines = [s.clone()[None, :] for s in n_seeds]

        logger.info("Multiple GPU tracking: Starting forward propagation for "
                    "the next {} streamlines.".format(len(lines)))
        tracking_info = self.propagator.prepare_forward(n_seeds)
        lines = self._propagate_multiple_lines(lines, tracking_info)

        if not self.track_forward_only:
            rej_idx = self.propagator.reject_streamlines_before_backward(lines)
            if rej_idx is not None and len(rej_idx) > 0:
                lines = [s for i, s in enumerate(lines) if i not in rej_idx]
                n_seeds = [s for i, s in enumerate(n_seeds) if i not in rej_idx]

            logger.info("   Starting backward propagation for the remaining "
                        "{} streamlines.".format(len(lines)))
            # Reversing:
            lines = [torch.flip(line, (0,)) for line in lines]

            tracking_info = self.propagator.prepare_backward(
                lines, forward_dir=tracking_info)

            lines = self._propagate_multiple_lines(lines, tracking_info)

        # Clean streamlines
        # Using max + 1 because we do one last step at the end.
        lengths = np.asarray([len(line) for line in lines])
        good_lengths = np.logical_and(self.min_nbr_pts <= lengths,
                                      lengths <= self.max_nbr_pts + 1)
        good_lengths, = np.where(good_lengths)
        clean_lines = [lines[i] for i in good_lengths]
        clean_seeds = [n_seeds[i] for i in good_lengths]
        return clean_lines, clean_seeds

    def _propagate_multiple_lines(self, lines, tracking_info):
        """
        Equivalent of super's() _propagate_lines() for for multiple tracking at
        the same time (meant to be used on GPU).
        """
        nb_streamlines = len(lines)

        # Monitoring
        invalid_direction_counts = np.zeros(nb_streamlines)
        final_tracking_info = [None] * nb_streamlines  # type: List[torch.Tensor]
        continuing_lines_rawidx = np.arange(nb_streamlines)

        # Will get the final lines when they are done.
        final_lines = [None] * nb_streamlines  # type: List[torch.Tensor]

        # `lines` will be updated at each loop to only contain the remaining
        # lines.
        all_lines_completed = False
        current_step = 0  # This count is reset for forward and backward
        while not all_lines_completed:
            current_step += 1
            nb_streamlines_left = len(lines)
            logging.debug("Propagation step #{}".format(current_step))

            n_new_pos, tracking_info, valid_dirs = self.propagator.propagate(
                lines, tracking_info)

            # Verifying and appending
            invalid_direction_counts[~valid_dirs.cpu().numpy()] += 1

            all_lines_completed = True
            continuing_lines = np.ones(nb_streamlines_left).astype(bool)
            for i in range(len(lines)):
                # In non-simultaneous, nbr pts is verified in the loop.
                # Here, during forward, we can do that, but during backward,
                # all streamlines have different lengths.
                propagation_can_continue = (
                        self._verify_stopping_criteria(
                            invalid_direction_counts[i],
                            n_new_pos[i].cpu().numpy()) and
                        len(lines[i]) <= self.max_nbr_pts)

                if propagation_can_continue:
                    all_lines_completed = False
                    lines[i] = torch.vstack((lines[i], n_new_pos[i]))
                else:
                    continuing_lines[i] = False
                    final_lines[continuing_lines_rawidx[i]] = lines[i]
                    final_tracking_info[continuing_lines_rawidx[i]] = \
                        tracking_info[i]

            # Keeping only remaining lines
            continuing_lines_subidx, = np.where(continuing_lines)
            continuing_lines_rawidx = continuing_lines_rawidx[continuing_lines]
            lines = [lines[i] for i in continuing_lines_subidx]
            tracking_info = [tracking_info[i] for i in continuing_lines_subidx]
            invalid_direction_counts = invalid_direction_counts[continuing_lines]

            # Update model if needed.
            self.propagator.multiple_lines_update(continuing_lines_subidx)

        assert len(lines) == 0
        assert len(continuing_lines_rawidx) == 0

        # Possible last step.
        final_lines = self.propagator.finalize_streamlines(
            final_lines, final_tracking_info, self.mask)

        return final_lines

    def _verify_stopping_criteria(self, invalid_direction_count, last_pos):

        # Checking number of consecutive invalid directions
        if invalid_direction_count > self.max_invalid_dirs:
            return False

        # Checking if out of bound
        if not self.mask.is_coordinate_in_bound(
                *last_pos, space=self.space, origin=self.origin):
            return False

        # Checking if out of mask
        if self.mask.get_value_at_coordinate(
                *last_pos, space=self.space, origin=self.origin) <= 0:
            return False

        return True
