# -*- coding: utf-8 -*-
from collections import defaultdict
import logging
from typing import Dict, List, Union, Tuple, Iterator

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft
import torch
import torch.multiprocessing
from torch.utils.data import Sampler

from dwi_ml.experiment_utils.prints import TqdmLoggingHandler
from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)

# For the batch sampler with inputs
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood

"""
                                Batch sampler

These classes defines how to sample the streamlines available in the
MultiSubjectData. 

AbstractBatchSampler:

- Defines the __iter__ method: 
    - Finds a list of streamlines ids and associated subj that you can later 
    load in your favorite way.
    
    - It is possible to restrict the number of subjects in a batch (and thus 
    the number of inputs to load associated with sampled streamlines), and to 
    reduce the number of time we need to load new data by using the same 
    subjects for a given number of "cycles".
    
- Define the load_batch method:
    - Loads the streamlines associated to sampled ids. Can resample them.
    
    - Performs data augmentation (on-the-fly to avoid having to multiply data
     on disk) (ex: splitting, reversing, adding noise).

    NOTE: Actual loaded batch size might be different than `batch_size`
    depending on chosen data augmentation. This sampler takes streamline
    cutting and resampling into consideration, and as such, will return more
    (or less) points than the provided `batch_size`.

----------
                        Implemented child classes

BatchStreamlinesSamplerOneInput:

- Redefines the load_batch method:
    - Now also loads the input data under each point of the streamline (and 
    possibly its neighborhood), for one input volume.
 
You are encouraged to contribute to dwi_ml by adding any child class here.

USAGE:
Can be used in a torch DataLoader. For instance:
        # Initialize dataset
        dataset = MultiSubjectDataset(...)
        dataset.load_training_data()
        # Initialize batch sampler
        batch_sampler = BatchSampler(...)
        # Use this in the dataloader
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                collate_fn=batch_sampler.load_batch)
"""


class AbstractBatchSampler(Sampler):
    def __init__(self, dataset: MultisubjectSubset,
                 streamline_group_name: str, chunk_size: int,
                 max_batch_size: int, rng: int,
                 nb_subjects_per_batch: int, cycles: int,
                 step_size: float, compress: bool,
                 split_ratio: float, noise_gaussian_size: float,
                 noise_gaussian_variability: float,
                 reverse_ratio: float):
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

        # Batch sampler's logging level can be changed separately from main
        # scripts.
        self.logger = logging.getLogger('batch_sampler_logger')
        self.logger.propagate = False
        self.logger.setLevel(logging.root.level)

    def make_logger_tqdm_fitted(self):
        """Possibility to use a tqdm-compatible logger in case the model
        is used through a tqdm progress bar."""
        self.logger.addHandler(TqdmLoggingHandler())

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
            'type': type(self),
            'noise_gaussian_size': self.noise_gaussian_size,
            'noise_gaussian_variability': self.noise_gaussian_variability,
            'reverse_ratio': self.reverse_ratio,
            'split_ratio': self.split_ratio,
            'step_size': self.step_size,
            'compress': self.compress,
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
        self.logger.debug("********* Entering batch sampler iteration!\n"
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
                self.logger.info("No streamlines remain for this epoch, "
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
            self.logger.debug('    Sampled subjects for this batch: {}'
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
                self.logger.debug(
                    "    Iteration for cycle # {}/{}."
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

                        if len(subbatch_rel_ids) == 0:
                            logging.warning(
                                "MAJOR WARNING. Got no streamline for this "
                                "subject in this batch, but there are "
                                "streamlines left. \nPossibly means that the "
                                "allowed batch size does not even allow one "
                                "streamline per batch. Check your batch size "
                                "choice!")

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

                self.logger.debug(
                    "    Finished loop on subjects. Now yielding.\n"
                    "    (If this was called through a dataloader, it should "
                    "start using load_batch and even training \n"
                    "     on this batch while we prepare a batch for the next "
                    "cycle, if any).")

                if len(batch_ids_per_subj) == 0:
                    self.logger.debug(
                        "No more streamlines remain in any of the "
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

        self.logger.debug("    Available streamlines for this subject: {}"
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
                nb_points = l_points[self.streamline_group_idx][
                    chosen_global_ids]
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
            self.logger.debug(
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
            (batch_streamlines, final_s_ids_per_subj)
        Where
            - batch_streamlines: list[np.array]
                The new streamlines after data augmentation
            - final_s_ids_per_subj: Dict[int, slice]
                The new streamline ids per subj in this augmented batch.
        """
        self.logger.debug("        Loading a batch of streamlines!")

        (batch_streamlines, final_s_ids_per_subj) = \
            self.streamlines_data_preparation(streamline_ids_per_subj)

        return batch_streamlines, final_s_ids_per_subj

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
            self.logger.debug("        => Subj: {}".format(subj))

            self.logger.debug(
                "          Processing data preparation for streamlines ids:\n"
                "{}".format(s_ids))

            subj_data = self.dataset.subjs_data_list.open_handle_and_getitem(
                subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # Get streamlines as sft
            sft = subj_sft_data.as_sft(s_ids)

            # Resampling streamlines to a fixed step size, if any
            self.logger.debug("            Resampling: {}"
                              .format(self.step_size))
            if self.step_size:
                if self.dataset.step_size == self.step_size:
                    self.logger.debug("Step size is the same as when creating "
                                      "the hdf5 dataset. Not resampling "
                                      "again.")
                else:
                    sft = resample_streamlines_step_size(
                        sft, step_size=self.step_size)

            # Compressing, if wanted.
            self.logger.debug(
                "            Compressing: {}".format(self.compress))
            if self.compress:
                sft = compress_sft(sft)

            # Adding noise to coordinates
            # Noise is considered in mm so we need to make sure the sft is in
            # rasmm space
            add_noise = bool(self.noise_gaussian_size and
                             self.noise_gaussian_size > 0)
            self.logger.debug("            Adding noise: {}".format(add_noise))
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
            self.logger.debug("            Splitting: {}".format(do_split))
            if do_split:
                all_ids = np.arange(len(sft))
                n_to_split = int(np.floor(len(sft) * self.split_ratio))
                split_ids = self.np_rng.choice(all_ids, size=n_to_split,
                                               replace=False)
                sft = split_streamlines(sft, self.np_rng, split_ids)

            # Reversing streamlines
            do_reverse = self.reverse_ratio and self.reverse_ratio > 0
            self.logger.debug("            Reversing: {}".format(do_reverse))
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
            # the underlying input(s). Sending to vox and to corner to
            # be able to use our trilinear interpolation
            sft.to_vox()
            sft.to_corner()
            batch_streamlines.extend(sft.streamlines)

            return batch_streamlines, final_s_ids_per_subj

    def project_specific_data_augmentation(self, sft: StatefulTractogram):
        """Please override in your child class if you want to do more than
        - reversing
        - adding noise
        - splitting."""
        self.logger.debug("            Project-specific data augmentation, if "
                          "any...")

        return sft


class BatchStreamlinesSamplerOneInput(AbstractBatchSampler):
    """
    Samples:
        input = one volume group
                (data underlying each point of the streamline)
                (possibly with its neighborhood)
        target = the whole streamlines as sequences.
    """
    def __init__(self, dataset: MultisubjectSubset,
                 streamline_group_name: str,
                 chunk_size: int, max_batch_size: int, rng: int,
                 nb_subjects_per_batch: int, cycles: int, compress: bool,
                 step_size: float, split_ratio: float,
                 noise_gaussian_size: float,
                 noise_gaussian_variability: float,
                 reverse_ratio: float, input_group_name, wait_for_gpu: bool,
                 neighborhood_points: np.ndarray):
        """
        Additional parameters compared to super:
        --------
        input_group_name: str
            Name of the input group in the hdf5 dataset.
        wait_for_gpu: bool
            If true, will not compute the inputs directly when using
            load_batch. User can call the compute_inputs method himself later
            on. Typically, Dataloader (who call load_batch) uses CPU.
        neighborhood_points: np.ndarray
            The list of neighborhood points (does not contain 0,0,0 point)
        """
        super().__init__(dataset, streamline_group_name, chunk_size,
                         max_batch_size, rng, nb_subjects_per_batch, cycles,
                         step_size, compress, split_ratio, noise_gaussian_size,
                         noise_gaussian_variability, reverse_ratio)

        # toDo. Would be more logical to send this as params when using
        #  load_batch as collate_fn in the Dataloader during training.
        #  Possible?
        self.wait_for_gpu = wait_for_gpu

        self.input_group_name = input_group_name
        self.neighborhood_points = neighborhood_points

        # Find group index in the data_source
        idx = self.dataset.volume_groups.index(input_group_name)
        self.input_group_idx = idx

    @property
    def params(self):
        p = super().params
        p.update({
            'input_group_name': self.input_group_name,
            'neighborhood_points': self.neighborhood_points,
            'wait_for_gpu': self.wait_for_gpu
        })
        return p

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]],
                   save_batch_input_mask: bool = False) \
            -> Union[Tuple[List, Dict],
                     Tuple[List, List, List]]:
        """
        Same as super but also interpolated the underlying inputs (if
        wanted) for all subjects in batch.

        Torch uses this function to process the data with the dataloader
        workers. To be used as collate_fn. This part is ran on CPU.

        With self.wait_for_gpu option: avoiding non-necessary operations in the
        batch sampler, computed on cpu: compute_inputs can be called later by
        the user.
        >> inputs = sampler.compute_inputs(batch_streamlines,
                                           streamline_ids_per_subj)

        Additional parameters compared to super:
        ----------
        save_batch_input_mask: bool
            Debugging purposes. Saves the input coordinates as a mask. Must be
            used together with wait_for_gpu=False. The inputs will be modified
            to a tuple containing the batch_streamlines (with wait_for_gpu set
            to False, the directions only would be returned), to compare the
            streamlines with masks.

        Returns
        -------
        If self.wait_for_gpu: same as super. Else, also returns
            batch_inputs : List of torch.Tensor
                Inputs volume loaded from the given group name. Data is
                flattened. Length of the list is the number of streamlines.
                Size of tensor at index i is [N_i-1, nb_features].
        """
        batch = super().load_batch(streamline_ids_per_subj)

        if self.wait_for_gpu:
            # Only returning the streamlines for now.
            # Note that this is the same as using data_preparation_cpu_step
            self.logger.debug("            Not loading the input data because "
                              "user prefers to do it later on GPU.")

            return batch
        else:
            # At this point batch_streamlines are in voxel space with
            # corner origin.
            batch_streamlines, final_s_ids_per_subj = batch

            # Get the inputs
            self.logger.debug("        Loading a batch of inputs!")
            batch_inputs = self.compute_inputs(batch_streamlines,
                                               final_s_ids_per_subj,
                                               save_batch_input_mask)

            return batch_streamlines, final_s_ids_per_subj, batch_inputs

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
            The streamlines (after data augmentation) in voxel space, with
            corner origin.
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
            Torch device.

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
            data_tensor = self.dataset.get_volume_verify_cache(
                subj, self.input_group_idx, device=device, non_blocking=True)

            # Prepare the volume data, possibly adding neighborhood
            # (Thus new coords_torch possibly contain the neighborhood points)
            # Coord_clipped contain the coords after interpolation
            # Trilinear interpolation uses origin=corner, vox space, but ok
            # because in load_batch, we use sft.to_vox and sft.to_corner
            # before adding streamline to batch.
            subj_x_data, coords_torch = interpolate_volume_in_neighborhood(
                data_tensor, flat_subj_x_coords, self.neighborhood_points,
                device)

            # Split the flattened signal back to streamlines
            lengths = [len(s) - 1 for s in batch_streamlines[y_ids]]
            subbatch_x_data = subj_x_data.split(lengths)
            batch_x_data.extend(subbatch_x_data)

            if save_batch_input_mask:
                print("DEBUGGING MODE. Returning batch_streamlines "
                      "and mask together with inputs.")

                # Clipping used coords (i.e. possibly with neighborhood)
                # outside volume
                lower = torch.as_tensor([0, 0, 0], device=device)
                upper = torch.as_tensor(data_tensor.shape[:3], device=device)
                upper -= 1
                coords_to_idx_clipped = torch.min(
                    torch.max(torch.floor(coords_torch).long(), lower),
                    upper)
                input_mask = torch.tensor(np.zeros(data_tensor.shape[0:3]))
                for s in range(len(coords_torch)):
                    input_mask.data[tuple(coords_to_idx_clipped[s, :])] = 1
                batch_input_masks.append(input_mask)

                return batch_input_masks, batch_x_data

        return batch_x_data
