# -*- coding: utf-8 -*-
import itertools
import logging
import multiprocessing
import os
import sys
import traceback
from typing import List

from dipy.tracking.streamlinespeed import compress_streamlines
import numpy as np
import torch
from dwi_ml.tracking.utils import prepare_step_size_vox
from torch import Tensor
from tqdm.contrib.logging import tqdm_logging_redirect

from scilpy.tracking.seed import SeedGenerator

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.direction_getter_models import \
    AbstractRegressionDG
from dwi_ml.models.main_models import ModelWithDirectionGetter, \
    MainModelOneInput
from dwi_ml.tracking.propagation import propagate_multiple_lines
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
                 min_len_mm: float, max_len_mm: float,
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

        if model.compress_lines:
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
        lines: List[List]
            List of lists of 3D positions (streamlines).
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
        streamlines: List[List]
            The successful streamlines.
        seeds: List[np.ndarray]
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
                    # Compressing. Threshold is in mm. Considering that we work
                    # in vox space, changing:
                    # Equivalent of sft.to_voxmm:
                    streamline *= self.seed_generator.voxres
                    compress_streamlines(streamline, self.compression_th)
                    # Equivalent of sft.to_vox:
                    streamline /= self.seed_generator.voxres

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
        Returns
        -------
        clean_lines: List[np.ndarray]
            The generated streamline for each seeding_pos.
        """
        torch.cuda.empty_cache()

        # List of list. Sending to Tensors.
        seeds = [torch.as_tensor(s, device=self.device, dtype=torch.float)
                 for s in seeds]
        lines = [s.clone()[None, :] for s in seeds]

        logger.debug("Starting forward")
        self.prepare_forward(seeds)
        lines = self._propagate_multiple_lines(lines)

        if not self.track_forward_only:
            logger.debug("Starting backward")
            lines, rej_idx = self.prepare_backward(lines)
            if rej_idx is not None and len(rej_idx) > 0:
                seeds = [s for i, s in enumerate(seeds) if i not in rej_idx]
            lines = self._propagate_multiple_lines(lines)

        # Clean streamlines
        # Max is already checked as stopping criteria.
        lengths = np.asarray([len(line) for line in lines])
        good_lengths, = np.where(self.min_nbr_pts <= lengths)
        clean_lines = [lines[i] for i in good_lengths]
        clean_seeds = [seeds[i] for i in good_lengths]

        return clean_lines, clean_seeds

    def _propagate_multiple_lines(self, lines: List[Tensor]):
        with torch.no_grad():
            return propagate_multiple_lines(
                lines, self.update_memory_after_removing_lines,
                self.get_next_dirs, self.theta, self.step_size,
                self.verify_opposite_direction, self.mask, self.max_nbr_pts,
                append_last_point=self.append_last_point,
                normalize_directions=self.normalize_directions)

    def get_next_dirs(self, lines: List[Tensor], n_last_pos: List[Tensor]):
        """
        Returns
        -------
        next_dirs: Tensor
            Input to model is a list of streamlines but output should be
            next_dirs = a Tensor of shape [nb_streamlines, 3].
        n_last_pos: List[Tensor]
        """
        inputs = self._prepare_inputs_at_pos(n_last_pos)

        model_outputs = self._call_model_forward(inputs, lines)

        next_dirs = self.model.get_tracking_directions(
            model_outputs, self.algo, self.eos_stopping_thresh)

        return next_dirs

    def prepare_forward(self, seeding_pos: List[Tensor]):
        """
        Prepare information necessary at the first point of the streamline
        for forward propagation.

        Returns
        -------
        initial_dirs: torch.Tensor
            Any tracking information necessary for the propagation.
        """
        # Our models should be able to get an initial direction even with no
        # starting information. Override if your model is different.
        pass

    def _prepare_inputs_at_pos(self, last_pos):
        raise NotImplementedError

    def _call_model_forward(self, inputs, lines):
        with self.grad_context:
            if self.model.forward_uses_streamlines:
                model_outputs = self.model(inputs, lines)
            else:
                model_outputs = self.model(inputs)
        return model_outputs

    def update_memory_after_removing_lines(self, can_continue: np.ndarray,
                                           new_stopping_lines_raw_idx: List):
        """
        In case you need to update your model's memory when removing a
        streamline.

        Params
        ------
        can_continue: np.ndarray
            Indexes of lines that are kept.
        new_stopping_lines_raw_idx: List
            Raw indexes of lines that can't continue in the initial batch.
        """
        pass

    def prepare_backward(self, lines: List[Tensor]):
        """
        Preparing backward.

        Parameters
        ----------
        lines: List[Tensor]
            Result from the forward tracking, reversed. Each tensor is of
            shape (nb_points, 3).

        Returns
        -------
        lines: List
            Updated streamlines: rejected faild ones + reversed.
        seeds: List
            Updated seeds: rejected faild ones.
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

        if len(lines) > 0:
            logger.debug("   Starting backward propagation for the remaining "
                         "{} streamlines.".format(len(lines)))

            # 2) If forward tracking succeeded, revert streamlines
            lines = [torch.flip(line, (0,)) for line in lines]

        return lines, rej_idx


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

    def prepare_backward(self, lines):
        # Reminder. Super:
        #  - Rejects lines that contain only the seed.
        #  - Reverts lines
        lines, rej_idx = super().prepare_backward(lines)

        # Not keeping the seed (input #0). Backward will start at that point,
        # and we will compute it again at _prepare_inputs_at_pos.
        # Also rejects failed inputs. They are already rejected in lines.
        self.input_memory_for_backward = \
            [s[1:, :] for i, s in enumerate(self.input_memory_for_backward)
             if i not in rej_idx]

        if self.append_last_point:
            # If the last direction was valid (i.e. not EOS), the last point
            # has been added to the streamline, but we never computed its
            # input. Computing that last additional input when
            # len(input) == len(lines) -1 (but we removed the seed point in input:)
            # len(input) == len(lines) - 2
            missing_one = [len(inp) == len(s) - 2 for inp, s in zip(
                self.input_memory_for_backward, lines)]
            idx_missing_one, = np.where(missing_one)
            if len(idx_missing_one) > 0:
                logger.debug("Computing last input for streamlines that had a "
                             "last valid step.")

                # Inputs: not reverted. Lines: reverted. Last input = line[0]
                last_inputs = self._prepare_inputs_at_pos(
                    [lines[i][0, :]for i in idx_missing_one])

                self.input_memory_for_backward = [
                    torch.vstack([self.input_memory_for_backward[i],
                                  last_inputs[where_first(idx_missing_one == i)]])
                    if missing_one[i] else self.input_memory_for_backward[i]
                    for i in range(len(lines))]

        # Reverting.
        self.input_memory = [torch.flip(line_input, dims=[0])
                             for line_input in self.input_memory_for_backward]
        self.input_memory_for_backward = 'deactivated'

        return lines, rej_idx

    def update_memory_after_removing_lines(
            self, can_continue: np.ndarray, new_stopping_lines_raw_idx: list):

        logger.debug("Updating memory after removing lines")
        if np.any(~can_continue):
            stopping_lines_sub_idx, = np.where(~can_continue)

            if self.input_memory_for_backward != 'deactivated':
                self.input_memory_for_backward: List
                # Forward! Saving removed inputs for later.
                for idx_r, idx_s in zip(new_stopping_lines_raw_idx,
                                        stopping_lines_sub_idx):
                    self.input_memory_for_backward[idx_r] = self.input_memory[idx_s]

            # Now removing from current inputs memory
            self.input_memory = [self.input_memory[i] for i in
                                 range(len(can_continue)) if can_continue[i]]

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
