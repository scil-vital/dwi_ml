# -*- coding: utf-8 -*-
from collections import defaultdict
import logging
from typing import Dict, List, Union, Iterator, Tuple

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft
import torch
import torch.multiprocessing
from torch.utils.data import Sampler

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)
from dwi_ml.models.main_models import MainModelAbstractNeighborsPreviousDirs

"""
BatchStreamlineSampler:
-----------------------
    Defines functions that can be of use for all models.

    Defines the __iter__ method: finds a list of streamlines ids and associated
         subj that you can later load in your favorite way. Most probably
         as done in the BatchStreamlinesSampler but we separated the
         implementation just in case.

    Defines the load_batch method that loads the streamlines and processes
    them (resampling + data augmentation).


You can then implement a child class that fits your needs. You are encouraged
to contribute to dwi_ml.

BatchStreamlinesSampler1IPV
---------------------------
    1IPV stands for 1 Input + Previous Dirs.

    x1: input = volume group named "input"
        (The underlying information under each point of the sampled
        streamlines)
    x2: prev_dirs = a list of n previous dirs
    y: target = the whole streamlines as sequences.
"""

"""
USAGE:
These batch samplers can be used in a torch DataLoader. For instance:
        # Initialize dataset
        dataset = MultiSubjectDataset(...)
        dataset.load_training_data()

        # Initialize batch sampler
        batch_sampler = BatchSampler(...)

        # Use this in the dataloader
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                collate_fn=batch_sampler.load_batch)
"""

"""
NOTE:
Currently, the __iter__ from the BatchSamplerAbstract samples complete
streamline ids, not just points. The MultiSubjectData would have to be
modified to find only some timesteps from a given streamline. But it is
probably easier to sample all points from each sampled streamline, and then
use them separatedly.
"""


class BatchStreamlinesSampler(Sampler):
    """
    This class defines how to sample the streamlines available in the
    MultiSubjectData. It is possible to restrict the number of subjects in a
    batch (and thus the number of inputs to load associated with sampled
    streamlines), and to reduce the number of time we need to load new data by
    using the same subjects for a given number of "cycles".

    NOTE: Actual batch sizes might be different than `batch_size` depending
    on chosen data augmentation. This sampler takes streamline cutting and
    resampling into consideration, and as such, will return more (or less)
    points than the provided `batch_size`.

    Data augmentation is done on-the-fly to avoid having to multiply data on
    disk.
    """

    def __init__(self, dataset: MultisubjectSubset,
                 streamline_group_name: str, chunk_size: int,
                 max_batch_size: int, rng: int,
                 nb_subjects_per_batch: int, cycles: int,
                 step_size: float, compress: bool,
                 split_ratio: float, noise_gaussian_size: float,
                 noise_gaussian_variability: float,
                 reverse_ratio: float, wait_for_gpu: bool,
                 normalize_directions: bool, **_):
        """
        Parameters
        ----------
        dataset : MultisubjectSubset
            Dataset to sample from.
        streamline_group_name: str
            The name of the group to use to load the sequences among all
            streamline_groups in the data_source.
        chunk_size: Number of streamlines to sample together while creating the
            batches. If the size of the streamlines is known in terms of
            number of points (resampling has been done, and no compressing is
            done in the batch sampler), we iteratively add chunk_size
            streamlines to the batch until the total number of sampled
            timepoint reaches the max_batch_size. Else, the total number of
            streamlines in the batch will be 1*chunk_size.
        max_batch_size : int
            Number of required points in a batch. Batches will be approximated
            as the final batch size depends on data augmentation (streamline
            cutting or resampling). Note that approximation of the number of
            streamlines to fit this batch size will depend on the type of
            step_size: fixed or compressed data.
        rng : int
            Seed for the random number generator.
        nb_subjects_per_batch : int
            Maximum number of subjects to be used in a single batch. Depending
            on the model, this can avoid loading too many input volumes at the
            same time, for example. If None, always sample from all subjects.
        cycles : int
            Used if `nb_subjects_per_batch` is given. Number of batches
            re-using the same subjects (and thus the same volumes) before
            sampling new ones.
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). If None, train on streamlines as they are.
            Note that you probably already fixed a step size when
            creating your dataset, but you could use a different one here if
            you wish. [None]
        compress: bool
            If true, compress streamlines. Cannot be used together with
            step_size. Once again, the choice can be different in the batch
            sampler than chosen when creating the hdf5.
        split_ratio : float
            DATA AUGMENTATION: Percentage of streamlines to randomly split
            into 2, in each batch (keeping both segments as two independent
            streamlines).
            The reason for cutting is to help the ML algorithm to track from
            the middle of WM by having already seen half-streamlines. If you
            are using interface seeding, this is not necessary.
        noise_gaussian_size : float
            DATA AUGMENTATION: Add random Gaussian noise to streamline
            coordinates with given variance. This corresponds to the std of the
            Gaussian. If step_size is not given, make sure it is smaller than
            your step size to avoid flipping direction. Ex, you could choose
            0.1 * step-size. Noise is truncated to +/- 2*noise_sigma and to
            +/- 0.5 * step-size (if given).
        noise_gaussian_variability: float
            DATA AUGMENTATION: If this is given, a variation is applied to the
            streamline_noise_gaussian_size to have more noisy streamlines and
            less noisy streamlines. This means that the real gaussian_size will
            be a random number between [size - variability, size + variability]
        reverse_ratio: float
            DATA AUGMENTATION: If set, reversed a part of the streamlines in
            the batch. You could want to reverse ALL your data and then use
            both the initial data and reversed data. But this would take twice
            the memory. If your ratio is, say, 0.5, streamlines have a 50%
            chance to be reversed. If you train for enough epochs, high chance
            that you will have used both directions of your streamline at least
            once. Default: 0.5.
            A way to absolutely ensure using both directions the same number of
            time, we could use a flag and at each epoch, reverse those with
            unreversed flag. But that adds a bool for each streamline in your
            dataset and probably not so useful.
        wait_for_gpu: bool
            If True, all computations that can be avoided in the batch sampler
            (which is computed on CPU) will be skipped and should be performed
            by the user later on GPU. Ex: computing directions from the
            streamline coordinates, computing input interpolation, etc.
        normalize_directions: bool
            If true, directions will be normalized. If the step size is fixed,
            it shouldn't make any difference. If streamlines are compressed,
            in theory you should normalize, but you could hope that not
            normalizing could give back to the algorithm a sense of distance
            between points.
        """
        super().__init__(dataset)  # This does nothing but python likes it.

        # Checking that batch_size is correct
        if (not isinstance(max_batch_size, int) or
                isinstance(max_batch_size, bool) or
                max_batch_size <= 0):
            raise ValueError("batch_size (i.e. number of total timesteps in "
                             "the batch) should be a positive integeral "
                             "value, but got batch_size={}"
                             .format(max_batch_size))

        # Checking that n_volumes was given if cycles was given
        if cycles and not nb_subjects_per_batch:
            raise ValueError("If `cycles` is defined, "
                             "`nb_subjects_per_batch` should be defined. Got: "
                             "nb_subjects_per_batch={}, cycles={}"
                             .format(nb_subjects_per_batch, cycles))

        # Batch sampler variables
        self.dataset = dataset
        self.streamline_group_name = streamline_group_name
        self.nb_subjects_per_batch = nb_subjects_per_batch
        self.cycles = cycles
        if step_size and compress:
            raise ValueError("You may choose either resampling or compressing,"
                             "but not both.")
        self.step_size = step_size
        self.compress = compress
        self.wait_for_gpu = wait_for_gpu
        self.normalize_directions = normalize_directions
        self.chunk_size = chunk_size
        self.max_batch_size = max_batch_size

        # Find idx of streamline group
        self.streamline_group_idx = self.dataset.streamline_groups.index(
            self.streamline_group_name)

        # Set random numbers
        self.rng = rng
        self.np_rng = np.random.RandomState(self.rng)
        torch.manual_seed(self.rng)  # Set torch seed

        # Data augmentation for streamlines:
        self.noise_gaussian_size = noise_gaussian_size
        self.noise_gaussian_variability = noise_gaussian_variability
        self.split_ratio = split_ratio
        self.reverse_ratio = reverse_ratio
        if self.split_ratio and not 0 <= self.split_ratio <= 1:
            raise ValueError('Split ratio must be a float between 0 and 1.')
        if self.reverse_ratio and not 0 <= self.reverse_ratio <= 1:
            raise ValueError('Reverse ration must be a float between 0 and 1')

        self.log = logging.getLogger()  # Gets the root logger

    @property
    def params(self):
        """
        All parameters. Contains at least all parameters that would be
        necessary to create this batch sampler again (except the dataset).
        """
        params = {
            'streamline_group_name': self.streamline_group_name,
            'max_batch_size': self.max_batch_size,
            'chunk_size': self.chunk_size,
            'rng': self.rng,
            'nb_subjects_per_batch': self.nb_subjects_per_batch,
            'cycles': self.cycles,
            'wait_for_gpu': self.wait_for_gpu,
            'type': type(self),
            'noise_gaussian_size': self.noise_gaussian_size,
            'noise_gaussian_variability': self.noise_gaussian_variability,
            'reverse_ratio': self.reverse_ratio,
            'split_ratio': self.split_ratio,
            'step_size': self.step_size,
            'compress': self.compress,
            'normalize_directions': self.normalize_directions
        }
        return params

    @property
    def states(self):
        states = {
            'rng_state': self.np_rng.get_state(),
        }
        return states

    def __iter__(self) -> Iterator[List[Tuple[int, list]]]:
        """
        Streamline sampling.

        First sample the subjects to be used from a given number of desired
        subjects, then sample streamline ids inside those volumes.

        Hint: To use this through the dataload, through a tqdm progress bar:
        with tqdm(dataloader) as pbar:
            train_iterator = enumerate(pbar)
            for batch_id, data in train_iterator:
                ...

        Returns
        -------
        batch_ids_per_subj : list[(int, list)]
            - The torch's dataloader will get this list and iterate on it, each
            time using __iter__ function of the dataset (a multisubjectSubset)
            to create a data and send it to the collate_fn (our load_batch).
            - This must be a list.
            - Inside the tuple, the int is the subject id and the list is
            the list of streamlines ids for this subject.
            - The list of streamline ids are relative ids inside each subject's
            tractogram).
        """
        # This is the list of all possible streamline ids
        global_streamlines_ids = np.arange(
            self.dataset.total_nb_streamlines[self.streamline_group_idx])
        ids_per_subjs = \
            self.dataset.streamline_ids_per_subj[self.streamline_group_idx]

        # This contains one bool per streamline:
        #   1 = this streamline has not been used yet.
        #   0 = this streamline has been used.
        global_unused_streamlines = np.ones_like(global_streamlines_ids)
        self.log.debug("********* Entering batch sampler iteration!\n"
                       "          Choosing out of {} possible streamlines"
                       .format(sum(global_unused_streamlines)))

        # This will continue "yielding" batches until it encounters a break.
        # (i.e. when all streamlines have been used)
        while True:
            # Weight subjects by their number of remaining streamlines
            streamlines_per_subj = np.array(
                [np.sum(global_unused_streamlines[subj_id_slice])
                 for _, subj_id_slice in ids_per_subjs.items()])
            assert (np.sum(streamlines_per_subj) ==
                    np.sum(global_unused_streamlines)), \
                "Unexpected error, the total number of streamlines per " \
                "subject does not correspond to the total number of " \
                "streamlines in the multisubject dataset. Error in " \
                "streamline ids?"

            # Stopping if all streamlines have been used
            if np.sum(streamlines_per_subj) == 0:
                self.log.info("No streamlines remain for this epoch, "
                              "stopping...")
                break

            # Choose subjects from which to sample streamlines for this batch
            if self.nb_subjects_per_batch:
                # Sampling first from subjects that were not seed a lot yet
                weights = streamlines_per_subj / np.sum(streamlines_per_subj)

                # Choosing only non-empty subjects
                nb_subjects = min(self.nb_subjects_per_batch,
                                  np.count_nonzero(weights))
                sampled_subjs = self.np_rng.choice(
                    np.arange(len(self.dataset.subjs_data_list)),
                    size=nb_subjects, replace=False, p=weights)
            else:
                # Sampling from all subjects
                sampled_subjs = ids_per_subjs.keys()
                nb_subjects = len(sampled_subjs)
            self.log.debug('    Sampled subjects for this batch: {}'
                           .format(sampled_subjs))

            batch_size_per_subj = self.max_batch_size / nb_subjects

            # Preparing to iterate on these chosen subjects for a predefined
            # number of cycles
            if self.cycles:
                iterator = range(self.cycles)
            else:
                # Infinite iterator, sampling from all subjects
                iterator = iter(int, 1)

            count_cycles = 0
            for _ in iterator:
                count_cycles += 1
                self.log.debug("    Iteration for cycle # {}/{}."
                               .format(count_cycles,
                                       self.cycles if self.cycles else 'inf'))

                # For each subject, randomly choose streamlines that have not
                # been chosen yet.
                batch_ids_per_subj = []
                for subj in sampled_subjs:
                    sampled_ids = []

                    # Get the global streamline ids corresponding to this
                    # subject
                    subj_slice = ids_per_subjs[subj]

                    # We will continue iterating on this subject until we
                    # break (i.e. when we reach the maximum batch size for this
                    # subject)
                    total_subj_batch_size = 0
                    while True:
                        (subbatch_global_ids, subbatch_rel_ids, subj_heaviness,
                         no_streamlines_left, reached_max) = \
                            self._prepare_subj_subbatch_ids(
                                subj_slice, global_unused_streamlines,
                                total_subj_batch_size, batch_size_per_subj)

                        if no_streamlines_left:
                            # No streamlines remaining. Get next subject.
                            break

                        # Mask the sampled streamlines
                        global_unused_streamlines[subbatch_global_ids] = 0

                        # Add sub-sample to sub's batch
                        sampled_ids.extend(subbatch_rel_ids)

                        # Continue?
                        if reached_max:
                            # Batch size reached for this subject. Get next
                            # subject.
                            break
                        else:
                            # Update heaviness and get a new chunk
                            total_subj_batch_size += subj_heaviness

                    # Append tuple (subj, list_sampled_ids) to the batch
                    batch_ids_per_subj.append((subj, sampled_ids))

                self.log.debug(
                    "    Finished loop on subjects. Now yielding.\n"
                    "    (If this was called through a dataloader, it should "
                    "start using load_batch and even training \n"
                    "     on this batch while we prepare a batch for the next "
                    "cycle, if any).")

                if len(batch_ids_per_subj) == 0:
                    self.log.debug("No more streamlines remain in any of the "
                                   "selected volumes! Restarting the batch "
                                   "sampler!")
                    break

                yield batch_ids_per_subj

            # Finished cycle. Will choose new subjs if the number of iterations
            # is not reached for this __iter__ call.

    def _prepare_subj_subbatch_ids(self, subj_slice, global_unused_streamlines,
                                   total_heaviness, max_heaviness):
        """
        Returns:
        (chosen_global_ids, chosen_relative_ids, no_streamlines_remaining,
        reached_max_heaviness)
        """
        no_streamlines_remaining = False
        reached_max_heaviness = False

        # Find streamlines that have not been used yet for this subj
        subj_unused_ids_in_global = np.flatnonzero(
            global_unused_streamlines[subj_slice]) + subj_slice.start

        self.log.info("    Available streamlines for this subject: {}"
                       .format(len(subj_unused_ids_in_global)))

        # No streamlines remain for this subject
        if len(subj_unused_ids_in_global) == 0:
            no_streamlines_remaining = True
            return [], [], no_streamlines_remaining, reached_max_heaviness

        # Sample a sub-batch of streamlines
        chosen_global_ids = self.np_rng.choice(subj_unused_ids_in_global,
                                               self.chunk_size)

        if (self.step_size or self.dataset.step_size) and not self.compress:
            # Relying on the lengths_mm info available in the MultiSubjectData
            # to be able to know the (eventual, if self.step_size) number of
            # time steps without loading the streamlines, particularly with the
            # lazy data.
            if self.step_size:
                l_mm = self.dataset.streamline_lengths_mm
                l_mm = l_mm[self.streamline_group_idx][chosen_global_ids]
                nb_points = l_mm / self.step_size
            else:
                l_points = self.dataset.streamline_lengths
                nb_points = l_points[self.streamline_group_idx][chosen_global_ids]
                # Should be equal to
                # nb_points = lengths_mm / self.dataset.step_size

            # If batch_size has been passed, taking a little less
            # streamlines for this last sub_batch.
            if total_heaviness + np.sum(nb_points) >= max_heaviness:
                cumulative_sum = np.cumsum(nb_points)
                selected = cumulative_sum < (max_heaviness - total_heaviness)
                chosen_global_ids = chosen_global_ids[selected]
                nb_points = nb_points[selected]
                reached_max_heaviness = True

            sample_heaviness = np.sum(nb_points)
            self.log.debug(
                "    Chunk_size was {} streamlines, but after verifying data "
                "heaviness in number of points (max batch size for this "
                "subj is {}), keeping only {} streamlines for a total of {}"
                "points."
                .format(self.chunk_size, max_heaviness,
                        len(chosen_global_ids), sample_heaviness))

        else:
            # Either we will compress data or we are taking the data as is
            # with no resampling: we have no way of knowing the final size of
            # data. We will simply take the given chunk of streamlines. Thus
            # stopping now. Setting sample_heaviness to max heaviness to stop
            # loop.
            sample_heaviness = None
            reached_max_heaviness = True

        # Fetch subject-relative ids
        chosen_relative_ids = list(chosen_global_ids - subj_slice.start)

        return (chosen_global_ids, chosen_relative_ids, sample_heaviness,
                no_streamlines_remaining, reached_max_heaviness)

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]]) \
            -> Union[Tuple[List, Dict], Tuple[List, List, List]]:
        """
        Fetches the chosen streamlines for all subjects in batch.
        Pocesses data augmentation.

        Torch uses this function to process the data with the dataloader
        parallel workers (on cpu). To be used as collate_fn.

        Parameters
        ----------
        streamline_ids_per_subj: List[Tuple[int, list]]
            The list of streamline ids for each subject (relative ids inside
            each subject's tractogram) for this batch.

        Returns
        -------
        If self.wait_for_gpu:
            (batch_streamlines, final_s_ids_per_subj)
        Else:
            (batch_streamlines, final_s_ids_per_subj, batch_directions)
        Where
            - batch_streamlines: list[np.array]
                The new streamlines after data augmentation
            - batch_directions: list[torch.Tensor]
                The (normalized) directions computed from each streamline.
            - final_s_ids_per_subj: Dict[int, slice]
                The new streamline ids per subj in this augmented batch.
        """
        self.log.debug("        Loading a batch of streamlines!")

        (batch_streamlines, final_s_ids_per_subj) = \
            self.streamlines_data_preparation(streamline_ids_per_subj)

        if self.wait_for_gpu:
            self.log.debug("            Not computing directions because user "
                           "prefers to do it later on GPU.")
            return batch_streamlines, final_s_ids_per_subj
        else:
            directions = self.compute_and_normalize_directions(
                batch_streamlines)
            return batch_streamlines, final_s_ids_per_subj, directions

    def set_log(self, log):
        """Possibility to pass a tqdm-compatible logger in case the dataloader
        is iterated through a tqdm progress bar. Note that, of course, log
        outputs will be confusing, particularly in debug mode, considering
        that the dataloader may use more than one method in parallel."""
        self.log = log

    def streamlines_data_preparation(
            self, streamline_ids_per_subj: List[Tuple[int, list]]):
        """
        Parameters
        -----------
        streamline_ids_per_subj: List[Tuple[int, list]]
            The list of streamlines for this batch, per subject (corresponding
            to the streamlines ids in each subject's tractogram).

        Returns
        -------
        batch_streamlines: List[np.array]
            The new streamlines after data augmentation
        final_s_ids_per_subj: Dict[int, slice]
            The new streamline ids per subj in this augmented batch.
        """

        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        final_s_ids_per_subj = defaultdict(slice)
        batch_streamlines = []
        for subj, s_ids in streamline_ids_per_subj:
            self.log.debug("        => Subj: {}".format(subj))

            self.log.debug(
                "          Processing data preparation for streamlines ids:\n"
                "{}".format(s_ids))

            subj_data = self.dataset.subjs_data_list.open_handle_and_getitem(
                subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # Get streamlines as sft
            sft = subj_sft_data.as_sft(s_ids)

            # Resampling streamlines to a fixed step size, if any
            self.log.debug("            Resampling: {}".format(self.step_size))
            if self.step_size:
                if self.dataset.step_size == self.step_size:
                    self.log.debug("Step size is the same as when creating "
                                   "the hdf5 dataset. Not resampling again.")
                else:
                    sft = resample_streamlines_step_size(
                        sft, step_size=self.step_size)

            # Compressing, if wanted.
            self.log.debug("            Compressing: {}".format(self.compress))
            if self.compress:
                sft = compress_sft(sft)

            # Adding noise to coordinates
            # Noise is considered in mm so we need to make sure the sft is in
            # rasmm space
            add_noise = bool(self.noise_gaussian_size and
                             self.noise_gaussian_size > 0)
            self.log.debug("            Adding noise: {}".format(add_noise))
            if add_noise:
                sft.to_rasmm()
                sft = add_noise_to_streamlines(sft,
                                               self.noise_gaussian_size,
                                               self.noise_gaussian_variability,
                                               self.np_rng, self.step_size)

            # Splitting streamlines
            # This increases the batch size, but does not change the total
            # length
            do_split = bool(self.split_ratio and self.split_ratio > 0)
            self.log.debug("            Splitting: {}".format(do_split))
            if do_split:
                all_ids = np.arange(len(sft))
                n_to_split = int(np.floor(len(sft) * self.split_ratio))
                split_ids = self.np_rng.choice(all_ids, size=n_to_split,
                                               replace=False)
                sft = split_streamlines(sft, self.np_rng, split_ids)

            # Reversing streamlines
            do_reverse = self.reverse_ratio and self.reverse_ratio > 0
            self.log.debug("            Reversing: {}".format(do_reverse))
            if do_reverse:
                ids = np.arange(len(sft))
                self.np_rng.shuffle(ids)
                reverse_ids = ids[:int(len(ids) * self.reverse_ratio)]
                sft = reverse_streamlines(sft, reverse_ids)

            # In case user wants to do more with its data.
            sft = self.project_specific_data_augmentation(sft)

            # Remember the indices of this subject's (augmented) streamlines
            ids_start = len(batch_streamlines)
            ids_end = ids_start + len(sft)
            final_s_ids_per_subj[subj] = slice(ids_start, ids_end)

            # Add all (augmented) streamlines to the batch
            # What we want is the streamline coordinates, to eventually get
            # the underlying input(s). Sending to vox
            sft.to_vox()
            batch_streamlines.extend(sft.streamlines)

            return batch_streamlines, final_s_ids_per_subj

    def project_specific_data_augmentation(self, sft: StatefulTractogram):
        """Please override in your child class if you want to do more than
        - reversing
        - adding noise
        - splitting."""
        self.log.debug("            Project-specific data augmentation, if "
                       "any...")

        return sft

    def compute_and_normalize_directions(self, batch_streamlines,
                                         device=torch.device('cpu')):
        """
        Computations that can be computed either right when loading batch with
        self.load_batch() if wait_for_gpu is false, or later when GPU is used
        if true.

        Params
        ------
        batch_streamlines: list[np.array]
                The streamlines (after data augmentation)
        device: torch device
            If this function is called by load_batch (through the Dataloader's
            collate_fn), device will be set to its default: 'cpu'. We do not
            want the dataloader to use GPU because each parallel worker would
            be sending data to GPU at the same time, with risks of errors.
            If self.wait_for_gpu, non-mendatory steps will be skipped. For this
            batch sampler: computing and normalizing directions. You can then
            compute them later, using the the cuda device.
            >> self.compute_and_normalize_directions(
                   self, batch_streamlines, device=torch.device('cuda')
        """
        self.log.debug("            Computing and normalizing directions ")

        # Getting directions
        batch_directions = [torch.as_tensor(s[1:] - s[:-1],
                                            dtype=torch.float32,
                                            device=device)
                            for s in batch_streamlines]

        # Normalization:
        if self.normalize_directions:
            batch_directions = [s / torch.sqrt(torch.sum(s ** 2, dim=-1,
                                                         keepdim=True))
                                for s in batch_directions]

        return batch_directions


class BatchStreamlinesSamplerWithInputs(BatchStreamlinesSampler):
    """
    This is used to load data under the form:

    x1: input = volume group named "input"
    x2: prev_dirs = a list of n previous dirs

    y: target = the whole streamlines as sequences.

    This is for instance the batch sampler used by Learn2Track and by
    Transformers.
    """
    def __init__(self, dataset: MultisubjectSubset,
                 streamline_group_name: str,
                 chunk_size: int, max_batch_size: int, rng: int,
                 nb_subjects_per_batch: int, cycles: int, compress: bool,
                 step_size: float, split_ratio: float,
                 noise_gaussian_size: float,
                 noise_gaussian_variability: float,
                 reverse_ratio: float, wait_for_gpu: bool,
                 normalize_directions: bool,
                 model: MainModelAbstractNeighborsPreviousDirs, **_):
        """
        Additional parameters compared to super:
        """
        super().__init__(dataset, streamline_group_name, chunk_size,
                         max_batch_size, rng, nb_subjects_per_batch, cycles,
                         step_size, compress, split_ratio, noise_gaussian_size,
                         noise_gaussian_variability, reverse_ratio,
                         wait_for_gpu, normalize_directions)

        # Find group index in the data_source
        idx = self.dataset.volume_groups.index(model.input_group_name)
        self.input_group_idx = idx

        self.model = model

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]],
                   save_batch_input_mask: bool = False) \
            -> Union[Tuple[List, Dict],
                     Tuple[List, List, List]]:
        """
        Fetches the chosen streamlines + underlying inputs + previous_dirs (if
        wanted) for all subjects in batch. Pocesses data augmentation.

        Torch uses this function to process the data with the dataloader
        workers. To be used as collate_fn. This part is ran on CPU.

        With self.wait_for_gpu option: avoiding non-necessary operations in the
        batch sampler, computed on cpu. Here, directions are not computed,
        and interpolation is not done. If you want to compute them later, use
        >> directions = sampler.compute_and_normalize_directions(
               batch_streamlines, streamline_ids_per_subj)
        >> inputs, previous_dirs = sampler.compute_interpolation(
               batch_streamlines, streamline_ids_per_subj, directions)

        Parameters
        ----------
        streamline_ids_per_subj: List[Tuple[int, list]]
            The list of streamline ids for each subject (relative ids inside
            each subject's tractogram).
        save_batch_input_mask: bool
            Debugging purposes. Saves the input coordinates as a mask. Must be
            used together with wait_for_gpu=False. The inputs will be modified
            to a tuple containing the batch_streamlines (with wait_for_gpu set
            to False, the directions only would be returned), to compare the
            streamlines with masks.

        Returns
        -------
        If self.wait_for_gpu:
            batch_streamlines : List of np.ndarray with shape (N_i,3)
                The streamlines coordinates in voxel space, ordered by subject.
            final_streamline_ids_per_subj : Dict[int, slice]
                A dictionary that maps each subject to the list of (processed)
                streamlines.
        else:
            batch_inputs : List of torch.Tensor
                Inputs volume loaded from the given group name. Data is
                flattened. Length of the list is the number of streamlines.
                Size of tensor at index i is [N_i-1, nb_features].
            batch_directions : List of torch.Tensors
                Streamline directions. Length of the list is the number of
                streamlines. Size of tensor at index i is [N_i-1, 3].
            previous_dirs: List of torch.Tensors
                Streamline previous directions (when begginning of streamline
                has been reached, "preceding" direction is [NaN, NaN, NaN].
                Length of the list is the number of streamlines. Size of tensor
                at index i is [N_i-1, 3*self.nb_previous_dirs].
        """
        batch = super().load_batch(streamline_ids_per_subj)

        if self.wait_for_gpu:
            # Only returning the streamlines for now.
            # Note that this is the same as using data_preparation_cpu_step
            self.log.debug("            Not loading the input data because "
                           "user prefers to do it later on GPU.")

            return batch
        else:
            batch_streamlines, final_s_ids_per_subj, batch_directions = \
                batch

            # Get the inputs
            self.log.debug("        Loading a batch of inputs!")
            batch_inputs = self.compute_inputs(batch_streamlines,
                                               final_s_ids_per_subj,
                                               save_batch_input_mask)

            # Get the previous dirs
            self.log.debug("        Computing previous dirs!")
            previous_dirs = self.compute_prev_dirs(batch_directions)

            return batch_inputs, batch_directions, previous_dirs

    def compute_inputs(self, batch_streamlines: List[np.ndarray],
                       streamline_ids_per_subj: Dict[int, slice],
                       save_batch_input_mask: bool = False,
                       device=torch.device('cpu')):
        """
        Get the DWI (depending on volume: as raw, SH, fODF, etc.) volume for
        each point in each streamline (+ depending on options: neighborhood,
        and preceding diretion)

        Params
        ------
        batch_streamlines: list[np.array]
            The streamlines (after data augmentation)
        streamline_ids_per_subj: Dict[int, slice]
            The ids corresponding to each subject (so we can load the
            associated subject's input volume).
        save_batch_input_mask: bool
            Debugging purposes. Saves the input coordinates as a mask. Must be
            used together with wait_for_gpu=False. The inputs will be modified
            to a tuple containing the batch_streamlines (with wait_for_gpu set
            to False, the directions only would be returned), to compare the
            streamlines with masks.
        device: torch device
            If this function is called by load_batch (through the Dataloader's
            collate_fn), device will be set to its default: 'cpu'. We do not
            want the dataloader to use GPU because each parallel worker would
            be sending data to GPU at the same time, with risks of errors.
            If self.wait_for_gpu, non-mendatory steps will be skipped. For this
            batch sampler: computing and normalizing directions. You can then
            compute them later, using the the cuda device.
            >> self.compute_and_normalize_directions(
                   self, batch_streamlines, device=torch.device('cuda'))
            >> self.compute_inputs(
                   self, batch_streamlines, streamline_ids_per_subj,
                   device=torch.device('cuda'))
            >> self.compute_prev_dirs(
                   self, batch_streamlines, device=torch.device('cuda'))

        Returns
        -------
        batch_x_data : List
            The list of (list of) inputs for each streamlines
        """
        batch_x_data = []
        batch_input_masks = []

        for subj, y_ids in streamline_ids_per_subj.items():
            # Flatten = concatenate signal for all streamlines to process
            # faster. We don't use the last coord because it is used only to
            # compute the last target direction, it's not really an input
            flat_subj_x_coords = np.concatenate(
                [s[:-1] for s in batch_streamlines[y_ids]], axis=0)

            # Getting the subject's volume and sending to CPU/GPU
            # If data is lazy, get volume from cache or send to cache if
            # it wasn't there yet.
            data_volume = self.dataset.get_volume_verify_cache(
                subj, self.input_group_idx, device=device, non_blocking=True)

            # Prepare the volume data, possibly adding neighborhood
            # (Thus new coords_torch possibly contain the neighborhood points)
            # Coord_clipped contain the coords after interpolation
            subj_x_data, coords_torch, coords_clipped = \
                self.model.prepare_inputs(data_volume, flat_subj_x_coords,
                                          device)

            # Split the flattened signal back to streamlines
            lengths = [len(s) - 1 for s in batch_streamlines[y_ids]]
            subbatch_x_data = subj_x_data.split(lengths)
            batch_x_data.extend(subbatch_x_data)

            if save_batch_input_mask:
                logging.debug("DEBUGGING MODE. Returning batch_streamlines "
                              "and mask together with inputs.")
                input_mask = torch.tensor(np.zeros(data_volume.shape[0:3]))
                for s in range(len(coords_torch)):
                    input_mask.data[tuple(coords_clipped[s, :])] = 1
                batch_input_masks.append(input_mask)

                return batch_streamlines, batch_input_masks, batch_x_data

        return batch_x_data

    def compute_prev_dirs(self, batch_directions, device=torch.device('cpu')):
        return self.model.prepare_previous_dirs(batch_directions, device)
