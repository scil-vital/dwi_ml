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

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.main_models import ModelWithDirectionGetter, \
    MainModelOneInput
from dwi_ml.tracking.tracking_mask import TrackingMask

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
                 min_nbr_pts: int, max_nbr_pts: int, max_invalid_dirs: int,
                 step_size: float, algo: str, theta: float,
                 verify_opposite_direction: bool,
                 normalize_directions: bool = True,
                 compression_th=0.1, nbr_processes=1, save_seeds=False,
                 rng_seed=1234, track_forward_only=False,
                 simultanenous_tracking: int = 1, use_gpu: bool = False,
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
        min_nbr_pts: int, minimal streamline length.
        max_nbr_pts: int, maximal streamline length.
        max_invalid_dirs: int
            Maximal invalid direction. If >1 and a direction is invalid (Ex: If
            EOS is chosen in a model), the streamline will continue straight
            until reaching the number of invalid values allowed.
        step_size: float, the step size for tracking, in voxel space.
        algo: str, 'det' or 'prob'
        theta: float
            Maximum angle (radians) allowed between two steps during sampling
            of the next direction.
        verify_opposite_direction: bool
            If true, during propagation, inverse the direction if opposite
            direction is better aligned (i.e. data model is symmetrical).
            If false, your model should learn to guess the correct direction
            as output.
        normalize_directions: bool, (if true, normalize directions).
        compression_th: float, compression threshold.
        nbr_processes: int, number of CPU processes.
        save_seeds: bool, wheter to save seeds in the tractogram.
        rng_seed: float, random seed.
        track_forward_only: bool.
        simultanenous_tracking: bool,
            If true, track multiple lines at the same time. Intended for GPU.
        use_gpu: bool
        """
        self.mask = mask
        self.seed_generator = seed_generator
        self.nbr_seeds = nbr_seeds
        self.min_nbr_pts = min_nbr_pts
        self.max_nbr_pts = max_nbr_pts
        self.max_invalid_dirs = torch.as_tensor(max_invalid_dirs)
        self.compression_th = compression_th
        self.save_seeds = save_seeds
        self.mmap_mode = None
        self.rng_seed = rng_seed
        self.track_forward_only = track_forward_only
        self.skip = 0  # Not supported yet.
        self.printing_frequency = 1
        self.device = None
        self.dataset = dataset
        self.step_size = step_size
        self.subj_idx = subj_idx
        self.model = model

        self.algo = algo
        if algo not in ['det', 'prob']:
            raise ValueError("Propagator's algo should be 'det' or 'prob'.")

        self.theta = theta
        self.normalize_directions = normalize_directions
        if not normalize_directions and step_size != 1:
            logging.warning("Tracker not normalizing directions obtained as "
                            "output from the model. Using a step size other "
                            "than 1 does not really make sense. You probably "
                            "want to advance of exactly 1 * output.")

        # If output is already normalized, no need to do it again.
        if 'regression' in self.model.direction_getter.key and \
                self.model.direction_getter.normalize_outputs == 1:
            self.normalize_directions = False

        # Contrary to super: normalize direction is optional
        self.normalize_directions = normalize_directions
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
            logging.warning("Minimum number of points cannot be 0. Changed to "
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

        self.simultanenous_tracking = simultanenous_tracking
        self.use_gpu = use_gpu
        if use_gpu:
            if torch.cuda.is_available():
                logging.info("We will be using GPU!")
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
        self.model.move_to(device=device)
        self.mask.move_to(device)
        self.max_invalid_dirs.to(device)

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
        
    def _gpu_simultanenous_tracking(self):
        """
        Creating all seeds at once and propagating all streamlines together.
        """
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

        logger.info("Multiple GPU tracking: Starting forward propagation for "
                    "the next {} streamlines.".format(len(lines)))
        initial_dirs = self.prepare_forward(seeds)
        lines = self._propagate_multiple_lines(lines, initial_dirs)

        if not self.track_forward_only:
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
        Equivalent of super's() _propagate_lines() for for multiple tracking at
        the same time (meant to be used on GPU).
        """
        nb_streamlines = len(lines)

        # Monitoring
        invalid_direction_counts = torch.zeros(nb_streamlines,
                                               device=self.device)
        continuing_lines_rawidx = np.arange(nb_streamlines)

        # Will get the final lines when they are done.
        final_lines = [None] * nb_streamlines  # type: List[torch.Tensor]

        # `lines` will be updated at each loop to only contain the remaining
        # lines.
        all_lines_completed = False
        current_step = 0  # This count is reset for forward and backward
        previous_dir = initial_dir
        while not all_lines_completed:
            current_step += 1
            logging.debug("Propagation step #{}".format(current_step))

            n_new_pos, previous_dir, valid_dirs = \
                self.take_one_step_or_go_straight(lines, previous_dir)

            # Verifying
            invalid_direction_counts[~valid_dirs] += 1
            can_continue = self._verify_stopping_criteria(
                invalid_direction_counts, n_new_pos, lines).cpu().numpy()

            # Saving finished streamlines
            new_stopping_lines_raw_idx = continuing_lines_rawidx[~can_continue]
            for i in range(len(lines)):
                if ~can_continue[i]:
                    final_lines[continuing_lines_rawidx[i]] = lines[i]

            # Keeping only remaining lines (adding last pos)
            lines = [torch.vstack((s, n_new_pos[i]))
                     for i, s in enumerate(lines) if can_continue[i]]
            previous_dir = previous_dir[can_continue]
            continuing_lines_rawidx = continuing_lines_rawidx[can_continue]
            invalid_direction_counts = invalid_direction_counts[can_continue]

            # Update model if needed.
            self.update_memory_after_removing_lines(
                can_continue, new_stopping_lines_raw_idx, nb_streamlines)

            all_lines_completed = ~np.any(can_continue)

        assert len(lines) == 0
        assert len(continuing_lines_rawidx) == 0
        assert len(final_lines) == nb_streamlines

        return final_lines

    def take_one_step_or_go_straight(self, lines, previous_dirs):
        """
        Params
        ------
        line: List[Tensor]
            For each streamline, tensor of shape (nb_points, 3).
        previous_dirs: Tensor(n, 3)
            Previous tracking directions. Can contain [NaN, NaN, NaN] for some
            points.

        Return
        ------
        n_new_pos: list[Tensor(3,)]
            The new segment position.
        next_dirs: list[Tensor(3,)]
            The new segment direction. None if no valid direction
            is found. Normalized if self.normalize_directions.
        valid_dirs: Tensor(nb_streamlines,)
            True if new_dir is valid.
        """
        last_pos = [line[-1, :] for line in lines]

        # Using the forward method to get the outputs.
        inputs = self._prepare_inputs_at_pos(last_pos)
        model_outputs = self._call_model_forward(inputs, lines)

        # Input to model is a list of streamlines but output should be
        # next_dirs = a tensor of shape [nb_streamlines, 3].
        next_dirs = self.model.get_tracking_directions(model_outputs,
                                                       self.algo)

        if self.normalize_directions:
            # Will divide along correct axis.
            next_dirs /= torch.linalg.norm(next_dirs, dim=-1)[:, None]

        next_dirs = self._verify_angle(next_dirs, previous_dirs)

        valid_dirs = ~torch.isnan(next_dirs[:, 0])
        next_dirs[~valid_dirs, :] = previous_dirs[~valid_dirs, :]

        n_new_pos = [last_pos[i] + self.step_size * next_dirs[i, :] for i in
                     range(len(last_pos))]

        return n_new_pos, next_dirs, valid_dirs

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
        cos_angle = np.minimum(np.maximum(-1.0, cos_angle.cpu()), 1.0)
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

        # 2) If forward tracking succeeded, revert streamlines
        logger.info("   Starting backward propagation for the remaining "
                    "{} streamlines.".format(len(lines)))
        # Reversing:
        lines = [torch.flip(line, (0,)) for line in lines]

        # 3) New v_in = last direction
        # Could be the foward_dir inverted, but in our case: always None.
        # (normalized if self.normalize_directions)
        backward_dir = [s[-1, :] - s[-2, :] for s in lines]
        backward_dir = torch.vstack(backward_dir)
        if self.normalize_directions:
            backward_dir /= torch.linalg.norm(backward_dir, dim=-1)[:, None]

        return lines, seeds, backward_dir, rej_idx

    def _verify_stopping_criteria(self, invalid_direction_count, n_last_pos,
                                  lines):

        n_last_pos = torch.vstack(n_last_pos)

        # Checking total length. During forward: all the same length. Not
        # during backward.
        can_continue = torch.as_tensor(
            np.asarray([len(s) for s in lines]) < self.max_nbr_pts,
            device=self.device)

        # Checking number of consecutive invalid directions
        can_continue = torch.logical_and(
            can_continue, torch.less_equal(invalid_direction_count,
                                           self.max_invalid_dirs))

        # Checking if out of bound using seeding mask
        can_continue = torch.logical_and(
            can_continue, self.mask.is_vox_corner_in_bound(n_last_pos))

        if self.mask.data is not None and torch.any(can_continue):
            # Checking if out of mask
            tmp = can_continue.clone()
            can_continue[tmp] = self.mask.is_in_mask(n_last_pos[tmp])

        return can_continue


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
        self.input_memory_for_backward = None

    def prepare_forward(self, seeding_pos):
        self.input_memory = []
        self.input_memory_for_backward = [None] * len(seeding_pos)
        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, lines, seeds, forward_dir=None):
        lines, seeds, backward_dir, idx = super().prepare_backward(
            lines, seeds, forward_dir)

        # Reject failed inputs
        self.input_memory_for_backward = \
            [s for i, s in enumerate(self.input_memory_for_backward)
             if i not in idx]

        # Not keeping the initial input point. Backward will start at
        # that point and will compute it again.
        self.input_memory = [
            torch.flip(line_input[1:, :], dims=[0])
            for line_input in self.input_memory_for_backward]
        self.input_memory_for_backward = None

        return lines, seeds, forward_dir, idx

    def update_memory_after_removing_lines(
            self, can_continue: np.ndarray, new_stopping_lines_raw_idx: list,
            batch_size: int):

        if np.any(~can_continue):
            stopping_lines_sub_idx, = np.where(~can_continue)

            if self.input_memory_for_backward is not None:
                # Forward! Saving removed inputs for later.
                for sr, ss in zip(new_stopping_lines_raw_idx,
                                  stopping_lines_sub_idx):
                    self.input_memory_for_backward[sr] = self.input_memory[
                        ss]

            # Now updating input memory
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
        inputs, _ = self.model.prepare_batch_one_input(
            n_pos, self.dataset, self.subj_idx, self.volume_group)

        return inputs
