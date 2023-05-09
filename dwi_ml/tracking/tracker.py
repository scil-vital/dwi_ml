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
from scilpy.tracking.seed import SeedGenerator
from tqdm.contrib.logging import tqdm_logging_redirect

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.direction_getter_models import \
    AbstractRegressionDG
from dwi_ml.models.main_models import ModelWithDirectionGetter, \
    MainModelOneInput
from dwi_ml.tracking.tracking_mask import TrackingMask
from dwi_ml.tracking.utils import prepare_step_size_vox

logger = logging.getLogger('tracker_logger')


class DWIMLAbstractTracker:
    """
    Contrary to scilpy: not performing one last step straight when the
    streamline is finished.

    Abstract version: Uses only the last coordinate of the streamlines in the
    model to get the next direction.
    """
    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
                 model: ModelWithDirectionGetter, mask: TrackingMask,
                 seed_generator: SeedGenerator, nbr_seeds: int,
                 min_len_mm: float, max_len_mm: float, max_invalid_dirs: int,
                 step_size_mm: float, algo: str, theta: float,
                 verify_opposite_direction: bool,
                 compression_th=0.1, nbr_processes=1, save_seeds=False,
                 rng_seed=1234, track_forward_only=False,
                 simultaneous_tracking: int = 1, use_gpu: bool = False,
                 append_last_point=True, eos_stopping_thresh=None,
                 log_level=logging.WARNING):
        """
        Parameters
        ----------
        dataset: MultisubjectSubset
            Loaded testing set. Must be lazy to allow multiprocessing.
            Multi-subject tracking not implemented; it should contain only
            one subject.
        subj_idx: int, subject used for tracking.
        model: ModelWithDirectionGetter, your torch model.
        mask: TrackingMask.
            Can contain no data, but requires its parameter dim to verify if
            out of bound.
        seed_generator: SeedGenerator
        nbr_seeds: int, nb seeds total to track.
        min_len_mm: float, minimal streamline length.
        max_len_mm: float, maximal streamline length.
        max_invalid_dirs: int
            Maximal invalid direction. If >0 and a direction is invalid (Ex: If
            EOS is chosen in a model, or if the angle is too sharp), the
            streamline will continue straight until reaching the number of
            invalid values allowed. Default: 0.
        step_size_mm: float, the step size for tracking, in millimiters.
        algo: str, 'det' or 'prob'
        theta: float
            Maximum angle (radians) allowed between two steps during sampling
            of the next direction.
        verify_opposite_direction: bool
            If true, during propagation, inverse the direction if opposite
            direction is better aligned (i.e. data model is symmetrical).
            If false, your model should learn to guess the correct direction
            as output.
        compression_th: float, compression threshold.
        nbr_processes: int, number of CPU processes.
        save_seeds: bool, whether to save seeds in the tractogram.
        rng_seed: float, random seed.
        track_forward_only: bool.
        simultaneous_tracking: bool,
            If true, track multiple lines at the same time. Intended for GPU.
        use_gpu: bool
        append_last_point: bool
            If true, keep the last point (the one out of the tracking mask
            that triggered the stopping criteria). Default in scilpy: true.
            Default in dipy: true. But this should be carefully thought.
        eos_stopping_thresh: float or 'max'
            Threshold for the EOS value to trigger a stopping criteria (if
            your model supports EOS). Default: 0.5
        """
        self.mask = mask
        self.seed_generator = seed_generator
        self.nbr_seeds = nbr_seeds
        self.max_invalid_dirs = max_invalid_dirs
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.mmap_mode = None
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.skip = 0  # Not supported yet.
        self.printing_frequency = 1
        self.device = None
        self.dataset = dataset
        self.subj_idx = subj_idx
        self.model = model
        self.model.set_context('tracking')
        self.append_last_point = append_last_point
        self.eos_stopping_thresh = eos_stopping_thresh

        if model.compress:
            logger.warning(
                "Careful! Model was trained on compressed streamlines. "
                "Tractography with a fixed step size could lead to weird "
                "results")
        if self.eos_stopping_thresh:
            if not model.direction_getter.add_eos:
                logger.warning("You have added a threshold for the EOS "
                               "stopping criterion, but your model does not "
                               "support EOS.")
            else:
                try:
                    self.eos_stopping_thresh = float(self.eos_stopping_thresh)
                except ValueError:
                    if not self.eos_stopping_thresh == 'max':
                        raise ValueError(
                            "eos stopping criteria should be either a float "
                            "or the string 'max', but we got {}"
                            .format(self.eos_stopping_thresh))
                if self.eos_stopping_thresh == 'max' and isinstance(
                        model.direction_getter,
                        AbstractRegressionDG):
                    raise ValueError("Regression's EOS is a tag, not a class. "
                                     "It does not support the criterion "
                                     "'max'.")
        elif model.direction_getter.add_eos:
            # Using default 0.5
            self.eos_stopping_thresh = 0.5

        if step_size_mm is None:
            if model.step_size is None:
                raise ValueError("Please specify the step_size to use with "
                                 "this model.")
            else:
                step_size_mm = model.step_size
        elif model.step_size and step_size_mm != model.step_size:
            logger.warning(
                "Careful! Model was trained on streamlines resampled to {}mm,"
                "but you are now tracking with {}mm step size!"
                .format(model.step_size, step_size_mm))

        step_size_vox, normalize_directions = prepare_step_size_vox(
            step_size_mm, seed_generator.voxres)

        self.step_size = step_size_vox
        self.max_nbr_pts = int(max_len_mm / step_size_mm)
        self.min_nbr_pts = max(int(min_len_mm / step_size_mm), 1)
        self.normalize_directions = normalize_directions

        self.algo = algo
        if algo not in ['det', 'prob']:
            raise ValueError("Tracker's algo should be 'det' or 'prob'.")

        self.theta = theta
        if not normalize_directions and step_size_vox != 1:
            logger.warning("Tracker not normalizing directions obtained as "
                           "output from the model. Using a step size other "
                           "than 1 does not really make sense. You probably "
                           "want to advance of exactly 1 * output.")

        # If output is already normalized, no need to do it again.
        if 'regression' in self.model.direction_getter.key and \
                self.model.direction_getter.normalize_outputs == 1:
            self.normalize_directions = False

        # Contrary to super: normalize direction is optional
        self.verify_opposite_direction = verify_opposite_direction

        # -------- Context
        # Uses torch's module eval(), which "turns off" the training mode.
        self.model.eval()
        self.grad_context = torch.no_grad()

        # Space and origin
        # torch trilinear interpolation uses origin='corner', space=vox.
        self.origin = Origin('corner')
        self.space = Space.VOX

        # Nb points
        if self.min_nbr_pts <= 0:
            logger.warning("Minimum number of points cannot be 0. Changed to "
                           "1.")
            self.min_nbr_pts = 1

        # Either GPU or multi-processes
        self.nbr_processes = self._set_nbr_processes(nbr_processes)
        if nbr_processes > 2:
            if not dataset.is_lazy:
                raise ValueError("Multiprocessing only works with lazy data.")
            if use_gpu:
                raise ValueError("You cannot use both multi-processes and "
                                 "gpu.")

        self.simultaneous_tracking = simultaneous_tracking
        self.use_gpu = use_gpu
        if use_gpu:
            if torch.cuda.is_available():
                logger.info("We will be using GPU!")
                device = torch.device('cuda')
            else:
                raise ValueError("You chose GPU (cuda) device but it is not "
                                 "available!")
        else:
            device = torch.device('cpu')
        self.move_to(device)

        logger.setLevel(log_level)

    def move_to(self, device):
        self.device = device
        self.model.move_to(device)
        self.mask.move_to(device)

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
                logger.warning("Cannot determine number of cpus: "
                               "nbr_processes set to 1.")
                nbr_processes = 1

        if nbr_processes > self.nbr_seeds:
            nbr_processes = self.nbr_seeds
            logger.debug("Setting number of processes to {} since there were "
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
        if self.simultaneous_tracking > 1:
            return self._gpu_simultaneous_tracking()
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

    def reset_data(self):
        if self.dataset.is_lazy:
            # Empty cache
            self.dataset.volume_cache_manager = None

            # Remove all handles
            self.dataset.close_all_handles()

    def _cpu_prepare_multiprocessing_pool(self):
        """
        Prepare multiprocessing pool. We do not need to deal with data
        management, contrary to scilpy.
        """
        # Clearing data from memory
        self.reset_data()

        pool = multiprocessing.Pool(self.nbr_processes)
        return pool

    def _cpu_reload_data_for_new_process(self):
        """Nothing to do here. hdf5 deals well with multi-processes."""

        logger.info("Preparing for process id {}".format(os.getpid()))
        self.reset_data()

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
            logger.error("Operation _cpu_tracking_sub() failed.")
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
                logger.info("Process {} (id {}): {} / {}"
                            .format(chunk_id, os.getpid(), s, chunk_size))

            seed = self.seed_generator.get_next_pos(
                random_generator, indices, first_seed_of_chunk + s)

            # Forward and backward tracking
            line = self._get_multiple_lines_both_directions([seed])[0][0]

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
        
    def _gpu_simultaneous_tracking(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
        random_generator, indices = self.seed_generator.init_generator(
            self.rng_seed, self.skip)

        seed_count = 0
        lines = []
        seeds = []
        with tqdm_logging_redirect(total=self.nbr_seeds, ncols=100) as pbar:
            while seed_count < self.nbr_seeds:
                nb_next_seeds = self.simultaneous_tracking
                if seed_count + nb_next_seeds > self.nbr_seeds:
                    nb_next_seeds = self.nbr_seeds - seed_count

                next_seeds = np.arange(seed_count, seed_count + nb_next_seeds)

                n_seeds = self.seed_generator.get_next_n_pos(
                    random_generator, indices, next_seeds)

                tmp_lines, tmp_seeds = \
                    self._get_multiple_lines_both_directions(n_seeds)
                pbar.update(nb_next_seeds)
                lines.extend([line.tolist() for line in tmp_lines])
                seeds.extend([s.tolist() for s in tmp_seeds])

                seed_count += nb_next_seeds

        return lines, seeds

    def _get_multiple_lines_both_directions(self, seeds: List[np.ndarray]):
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
        seeds = [torch.as_tensor(s, device=self.device, dtype=torch.float)
                 for s in seeds]
        lines = [s.clone()[None, :] for s in seeds]

        logger.debug("Starting forward")
        initial_dirs = self.prepare_forward(seeds)
        lines = self._propagate_multiple_lines(lines, initial_dirs)

        if not self.track_forward_only:
            logger.debug("Starting backward")
            lines, seeds, backward_dir, _ = self.prepare_backward(
                lines, seeds, initial_dirs)

            lines = self._propagate_multiple_lines(lines, backward_dir)

        # Clean streamlines
        # Max is already checked as stopping criteria.
        lengths = np.asarray([len(line) for line in lines])
        good_lengths, = np.where(self.min_nbr_pts <= lengths)
        clean_lines = [lines[i] for i in good_lengths]
        clean_seeds = [seeds[i] for i in good_lengths]

        return clean_lines, clean_seeds

    def prepare_forward(self, seeding_pos):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation.

        Parameters
        ----------
        seeding_pos: Tensor or List(Tensor)
            The 3D coordinates or, for simultaneous tracking, list of 3D
            coordinates.

        Returns
        -------
        initial_dirs: torch.Tensor
            Any tracking information necessary for the propagation.
        """
        # Our models should be able to get an initial direction even with no
        # starting information. Override if your model is different.
        empty_pos = torch.full((3,), fill_value=torch.nan, device=self.device)
        forward_dirs = torch.tile(empty_pos, (len(seeding_pos), 1))

        return forward_dirs

    def _propagate_multiple_lines(self, lines, initial_dir):
        """
        Equivalent of super's() _propagate_lines() for multiple tracking at
        the same time (meant to be used on GPU).
        """
        nb_streamlines = len(lines)

        # Monitoring
        invalid_direction_counts = np.zeros(nb_streamlines)
        continuing_lines_rawidx = np.arange(nb_streamlines)

        # Will get the final lines when they are done.
        final_lines = [None] * nb_streamlines  # type: List[torch.Tensor]

        # `lines` will be updated at each loop to only contain the remaining
        # lines.
        all_lines_completed = False
        previous_dir = initial_dir
        while not all_lines_completed:
            n_new_pos, previous_dir, invalid_dirs = \
                self.take_one_step_or_go_straight(lines, previous_dir)

            # If invalid direction (ex: angle or EOS), stop now.
            invalid_direction_counts[invalid_dirs] += 1
            breaking_now = invalid_direction_counts > self.max_invalid_dirs
            if sum(breaking_now) > 0:
                logger.debug("{} streamlines with invalid directions for more "
                             "than allowed {} points."
                             .format(sum(breaking_now), self.max_invalid_dirs))

            # For other streamlines: verifying but appending only if option is
            # chosen.
            break_with_appending = self._verify_stopping_criteria(
                n_new_pos, lines)

            if self.append_last_point:
                # Appending last point only to streamlines not breaking now.
                # (i.e. wrong angle or NaN direction) more than
                # max invalid dirs.
                lines = [torch.vstack((s, n_new_pos[i, :])) if ~breaking_now[i]
                         else s for i, s in enumerate(lines)]
                breaking_now = np.logical_or(break_with_appending,
                                             breaking_now)
                can_continue = ~breaking_now
            else:
                # Appending last point only to continuing streamlines
                breaking_now = np.logical_or(break_with_appending,
                                             breaking_now)
                can_continue = ~breaking_now
                lines = [torch.vstack((s, n_new_pos[i, :])) if can_continue[i]
                         else s for i, s in enumerate(lines)]

            # Saving finished streamlines
            idx_stop, = np.where(breaking_now)
            for i in idx_stop:
                final_lines[continuing_lines_rawidx[i]] = lines[i]

            # Update model if needed.
            if np.any(breaking_now):
                new_stopping_lines_raw_idx = continuing_lines_rawidx[
                    ~can_continue]
                self.update_memory_after_removing_lines(
                    can_continue, new_stopping_lines_raw_idx, nb_streamlines)

            # Keeping only remaining lines.
            if np.any(can_continue):
                lines = [s for i, s in enumerate(lines) if can_continue[i]]
                previous_dir = previous_dir[can_continue, :]
                continuing_lines_rawidx = continuing_lines_rawidx[can_continue]
                invalid_direction_counts = invalid_direction_counts[can_continue]
            else:
                all_lines_completed = True

        assert not np.any([line is None for line in final_lines])

        return final_lines

    def take_one_step_or_go_straight(self, lines, previous_dirs):
        """
        Finds the next direction. If no valid direction is found (invalid = if
        the model returns NaN, ex if EOS is used, or if the angle is too
        sharp). Then, the previous direction is copied but the list of invalid
        directions is returned.

        Params
        ------
        line: List[Tensor]
            For each streamline, tensor of shape (nb_points, 3).
        previous_dirs: Tensor(n, 3)
            Previous tracking directions. Can contain [NaN, NaN, NaN] for some
            points.

        Return
        ------
        n_new_pos: Tensor(n, 3)
            The new positions.
        next_dirs: Tensor(n, 3)
            The new segment direction. The previous direction is copied if no
            valid direction is found. Normalized if self.normalize_directions.
        invalid_dirs: ndarray(n, )
            True if new_dir is invalid.
        """
        last_pos = [line[-1, :] for line in lines]

        # Using the forward method to get the outputs.
        inputs = self._prepare_inputs_at_pos(last_pos)
        model_outputs = self._call_model_forward(inputs, lines)

        # Input to model is a list of streamlines but output should be
        # next_dirs = a tensor of shape [nb_streamlines, 3].
        next_dirs = self.model.get_tracking_directions(
            model_outputs, self.algo, self.eos_stopping_thresh)

        if self.normalize_directions:
            # Will divide along correct axis.
            next_dirs /= torch.linalg.norm(next_dirs, dim=-1)[:, None]

        next_dirs = self._verify_angle(next_dirs, previous_dirs)

        # Copy previous dirs for invalid directions
        invalid_dirs = torch.isnan(next_dirs[:, 0])
        next_dirs[invalid_dirs, :] = previous_dirs[invalid_dirs, :]

        # Get new positions
        last_pos = torch.vstack(last_pos)
        n_new_pos = last_pos + self.step_size * next_dirs

        return n_new_pos, next_dirs, invalid_dirs.cpu().numpy()

    def _prepare_inputs_at_pos(self, last_pos):
        raise NotImplementedError

    def _call_model_forward(self, inputs, lines):
        with self.grad_context:
            if self.model.forward_uses_streamlines:
                model_outputs = self.model(inputs, lines)
            else:
                model_outputs = self.model(inputs)
        return model_outputs

    def _verify_angle(self, next_dirs: torch.Tensor,
                      previous_dirs: torch.Tensor):
        # toDo could we find a better solution for proba tracking?
        #  Resampling until angle < theta? Easy on the sphere (restrain
        #  probas on the sphere pics inside a cone theta) but what do we do
        #  for other models? Ex: For the Gaussian direction getter?
        if self.normalize_directions:
            # Already normalized
            cos_angle = torch.sum(next_dirs * previous_dirs, dim=1)
        else:
            norm1 = torch.linalg.norm(next_dirs, dim=-1)
            norm2 = torch.linalg.norm(previous_dirs, dim=-1)
            cos_angle = torch.sum(
                torch.div(next_dirs, norm1[:, None]) *
                torch.div(previous_dirs, norm2[:, None]), dim=1)

        # Resolving numerical instabilities:
        # (Converts angle to numpy)
        # Torch does not have a min() for tensor vs scalar. Using np. Ok,
        # small step.
        one = torch.ones(1, device=self.device)
        cos_angle = torch.minimum(torch.maximum(-one, cos_angle), one)
        angle = torch.arccos(cos_angle)

        if self.verify_opposite_direction:
            mask_angle = angle > np.pi / 2  # 90 degrees
            angle[mask_angle] = np.mod(angle[mask_angle] + np.pi, 2*np.pi)
            next_dirs[mask_angle] = - next_dirs[mask_angle]

        mask_angle = angle > self.theta
        next_dirs[mask_angle] = torch.full((3,), fill_value=torch.nan,
                                           device=self.device)

        return next_dirs

    def update_memory_after_removing_lines(
            self, can_continue: np.ndarray, new_stopping_lines_raw_idx: list,
            batch_size: int):
        """
        In case you need to update your model's memory when removing a
        streamline.
        """
        pass

    def prepare_backward(self, lines, seeds, forward_dir=None):
        """
        Preparing backward.

        Parameters
        ----------
        lines: List[Tensor]
            Result from the forward tracking, reversed. Each tensor is of
            shape (nb_points, 3).
        seeds: List[ndarray]
            List of seeds.
        forward_dir: Tensor
            First direction chosen at the forward step. Not used here but
            overwrite if your model needs it.

        Returns
        -------
        lines: List
            Updated streamlines: rejected faild ones + reversed.
        seeds: List
            Updated seeds: rejected faild ones.
        backward_dir: Tensor
            Initialization direction during forward tracking.
        idx: List
            List of rejected indices
        """
        # 1) Rejecting streamlines of length 1 (= the seed only). The backward
        # would produce the same result. Overwrite if your model can change
        # results when called twice at the same point.
        lengths = np.asarray([len(s) for s in lines])
        rej_idx, = np.where(lengths == 1)

        if rej_idx is not None and len(rej_idx) > 0:
            lines = [s for i, s in enumerate(lines) if i not in rej_idx]
            seeds = [s for i, s in enumerate(seeds) if i not in rej_idx]

        if len(lines) > 0:
            logger.debug("   Starting backward propagation for the remaining "
                         "{} streamlines.".format(len(lines)))

            # 2) If forward tracking succeeded, revert streamlines
            lines = [torch.flip(line, (0,)) for line in lines]

            # 3) New v_in = last direction
            # (normalized if self.normalize_directions)
            backward_dir = [s[-1, :] - s[-2, :] for s in lines]
            backward_dir = torch.vstack(backward_dir)
            if self.normalize_directions:
                backward_dir /= torch.linalg.norm(backward_dir, dim=-1)[:, None]
        else:
            backward_dir = []

        return lines, seeds, backward_dir, rej_idx

    def _verify_stopping_criteria(self, n_last_pos, lines):

        # Checking total length. During forward: all the same length. Not
        # during backward.
        stopping = np.asarray([len(s) for s in lines]) == self.max_nbr_pts
        if sum(stopping) > 0:
            logger.debug("{} streamlines stopping after reaching max nb "
                         "points ({})".format(sum(stopping), self.max_nbr_pts))

        # Checking if out of bound using seeding mask
        out_of_mask = ~self.mask.is_vox_corner_in_bound(n_last_pos).cpu().numpy()
        if sum(out_of_mask) > 0:
            logger.debug("{} streamlines stopping out of bounds."
                         .format(sum(out_of_mask)))
        stopping = np.logical_or(stopping, out_of_mask)

        if self.mask.data is not None and not np.all(stopping):
            # Checking if out of mask
            # Avoid interpolation for points that we already know can't
            # continue.
            still_on = ~stopping

            out_of_mask = ~self.mask.is_in_mask(n_last_pos[still_on]).cpu().numpy()
            if sum(out_of_mask) > 0:
                logger.debug("{} streamlines stopping out of mask."
                             .format(sum(out_of_mask)))
            stopping[still_on] = out_of_mask

        return stopping


class DWIMLTrackerFromWholeStreamline(DWIMLAbstractTracker):
    """
    As compared to the general tracker, here, we need to send the
    whole streamline to the model in order to generate the next point's
    position. As it is the tracker's job to generally manage the memory of
    streamlines, we do not have access to these values. We need to copy
    them in memory here as long as the streamline is not finished being
    tracked.

    Parameters
    ----------
    use_input_memory: bool
        Remember the input value(s) at each point (in addition to the
        streamline itself, i.e. the coordinates, always saved). Warning:
        could be heavier in memory. Default: False.
    """
    def __init__(self, **kw):
        super().__init__(**kw)

        # List of inputs, as formatted by the model.
        self.input_memory = []

        # For the backward: either we recompute all inputs, or we need to
        # remember them during forward everytime a streamline is finished.
        self.input_memory_for_backward = 'deactivated'

    def prepare_forward(self, seeding_pos):
        self.input_memory = []
        self.input_memory_for_backward = [None] * len(seeding_pos)
        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, lines, seeds, forward_dir=None):

        # Reminder. Super:
        #  - Rejects lines that contain only the seed.
        #  - Reverts lines
        lines, seeds, backward_dir, rej_idx = super().prepare_backward(
            lines, seeds, forward_dir)

        # Not keeping the seed (input #0). Backward will start at that point,
        # and we will compute it again at _prepare_inputs_at_pos.
        # Also rejects failed inputs. They are already rejected in lines.
        self.input_memory_for_backward = \
            [s[1:, :] for i, s in enumerate(self.input_memory_for_backward)
             if i not in rej_idx]

        # If the last direction was valid (i.e. not EOS), the last point has
        # been added to the streamline, but we never computed its input.
        # Computing that last additional input when
        # len(input) < len(lines)  (but we removed the seed point so:)
        # len(input) < len(lines) - 1
        missing_one = [len(inp) < len(s) - 1 for inp, s in zip(
            self.input_memory_for_backward, lines)]
        idx_missing_one, = np.where(missing_one)
        if len(idx_missing_one) > 0:
            logger.debug("Computing last input for streamlines that had a "
                         "last valid step.")
            last_inputs = self._prepare_inputs_at_pos(
                [lines[i][0, :] for i in idx_missing_one])
            self.input_memory_for_backward = [
                torch.vstack([self.input_memory_for_backward[i],
                              last_inputs[where_first(idx_missing_one == i)]])
                if missing_one[i] else self.input_memory_for_backward[i]
                for i in range(len(missing_one))]

        # Reverting.
        self.input_memory = [
            torch.flip(line_input, dims=[0])
            for line_input in self.input_memory_for_backward]
        self.input_memory_for_backward = 'deactivated'

        return lines, seeds, forward_dir, rej_idx

    def update_memory_after_removing_lines(
            self, can_continue: np.ndarray, new_stopping_lines_raw_idx: list,
            batch_size: int):

        logger.debug("Updating memory after removing lines")
        if np.any(~can_continue):
            stopping_lines_sub_idx, = np.where(~can_continue)

            if self.input_memory_for_backward != 'deactivated':
                # Forward! Saving removed inputs for later.
                for idx_r, idx_s in zip(new_stopping_lines_raw_idx,
                                        stopping_lines_sub_idx):
                    self.input_memory_for_backward[idx_r] = self.input_memory[idx_s]

            # Now removing from current inputs memory
            self.input_memory = [self.input_memory[i] for i in
                                 range(len(can_continue)) if
                                 can_continue[i]]

    def _call_model_forward(self, inputs, lines):

        # Adding the current input to the input memory
        if len(self.input_memory) == 0:
            self.input_memory = inputs
        else:
            # If they all had the same lengths we could concatenate
            # everything. But during backward, they don't.
            self.input_memory = \
                [torch.cat((self.input_memory[i], inputs[i]), dim=0)
                 for i in range(len(self.input_memory))]

        return super()._call_model_forward(self.input_memory, lines)


class DWIMLTrackerOneInput(DWIMLAbstractTracker):
    """
    This version is for when data is represented as:
    x: one volume input (formatted through the model, possibly with
       neighborhood).

    This is in fact similar to our batch sampler (with inputs): it needs to get
    the data points from the volume (+possibly add a neighborhood) and
    interpolate the data.
    """
    model: MainModelOneInput

    def __init__(self, input_volume_group: str, **kw):
        """
        Params
        ------
        input_volume_group: str
            The volume group to use as input in the model.
        """
        super().__init__(**kw)

        # During tracking: always computing all inputs, contrary to during
        # training. Telling model how to format input batch.
        self.model.skip_input_as_last_point = False

        # Find group index in the data
        self.volume_group = self.dataset.volume_groups.index(
            input_volume_group)

        # To help prepare the inputs
        self.volume_group_str = input_volume_group

        if self.dataset.is_lazy and self.dataset.cache_size == 0:
            logger.warning("With lazy data and multiprocessing, you should "
                           "not keep cache size to zero. Data would be "
                           "loaded again at each propagation step!")

    def _prepare_inputs_at_pos(self, n_pos):
        """
        Prepare inputs at current position: get the volume and interpolate at
        current coordinate (possibly get the neighborhood coordinates too).

        Params
        ------
        n_pos: List[Tensor(1, 3)]
            List of n "streamlines" composed of one point.
        """
        n_pos = [pos[None, :] for pos in n_pos]
        return self.model.prepare_batch_one_input(
            n_pos, self.dataset, self.subj_idx, self.volume_group)


def where_first(array):
    w, = np.where(array)
    return w[0]
