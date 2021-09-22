# -*- coding: utf-8 -*-
from collections import defaultdict
import logging
from typing import Dict, List, Union, Iterable, Iterator, Tuple

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
import torch
import torch.multiprocessing
from torch.utils.data import Sampler

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.data.processing.space.neighborhood import (
    extend_coordinates_with_neighborhood,
    prepare_neighborhood_information)
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)
from dwi_ml.data.processing.volume.interpolation import (
    torch_trilinear_interpolation)

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
    1IPV standard for 1 Input + Previous Dirs.

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

# Number of streamlines sampled at the same for a given subject (then, we
# evaluate if the batch_size has been reached. No need to sample only one
# streamline at the time!
CHUNK_SIZE = 256


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
                 streamline_group_name: str, batch_size: int, rng: int,
                 nb_subjects_per_batch: int, cycles: int,
                 step_size: float, neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, Iterable[float], None],
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
        batch_size : int
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
            (in mm). If None, train on streamlines as they are (ex,
            compressed). Note that you probably already fixed a step size when
            creating your dataset, but you could use a different one here if
            you wish. [None]
        neighborhood_type: str
            The type of neighborhood to add. One of 'axes', 'grid' or None. If
            None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). (Can be none)
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
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
            If set, all computations that can be avoided in the batch sampler
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
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size <= 0):
            raise ValueError("batch_size should be a positive integeral "
                             "value, but got batch_size={}".format(batch_size))

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
        self.step_size = step_size
        self.wait_for_gpu = wait_for_gpu
        self.normalize_directions = normalize_directions

        # Find idx of streamline group
        self.streamline_group = self.dataset.streamline_groups.index(
            self.streamline_group_name)

        # Set random numbers
        self.rng = rng
        self.np_rng = np.random.RandomState(self.rng)
        torch.manual_seed(self.rng)  # Set torch seed

        # Batch size computation:
        # If self.step_size: we can't rely on the current number of time steps
        # because we will resample streamlines. Relying on the lenths_mm info
        # available in the MultiSubjectData to be able to know the (eventual)
        # number of time steps without loading the streamlines, particularly
        # with the lazy data.
        # If self.step_size is None: not resampling. Then we can get the real
        # number of time steps in the streamlines, also already available in
        # the MultiSubjectData without loading the streamlines.
        if self.step_size:
            # batch size is in mm
            self.batch_size = self.step_size * batch_size
        else:
            # batch size is in number of points.
            self.batch_size = batch_size

        # Data augmentation for streamlines:
        self.noise_gaussian_size = noise_gaussian_size
        self.noise_gaussian_variability = noise_gaussian_variability
        self.split_ratio = split_ratio
        self.reverse_ratio = reverse_ratio
        if self.split_ratio and not 0 <= self.split_ratio <= 1:
            raise ValueError('Split ratio must be a float between 0 and 1.')
        if self.reverse_ratio and not 0 <= self.reverse_ratio <= 1:
            raise ValueError('Reverse ration must be a float between 0 and 1')

        # Preparing the neighborhood for use by the child classes
        if neighborhood_type and not (neighborhood_type == 'axes' or
                                      neighborhood_type == 'grid'):
            raise ValueError("neighborhood type must be either 'axes', 'grid' "
                             "or None!")
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius
        if self.neighborhood_type is None and neighborhood_radius:
            logging.warning("You have chosen not to add a neighborhood (value "
                            "None), but you have given a neighborhood radius. "
                            "Discarded.")
        if self.neighborhood_type and neighborhood_radius is None:
            raise ValueError("You must provide neighborhood radius to add a "
                             "neighborhood.")
        self.neighborhood_points = prepare_neighborhood_information(
            neighborhood_type, neighborhood_radius)

        self.log = logging.getLogger()  # Gets the root logger

    @property
    def hyperparameters(self):
        hyperparameters = {
            'neighborhood_radius': self.neighborhood_radius,
            'nb_neighbors': len(self.neighborhood_points) if
            self.neighborhood_points else None,
            'noise_gaussian_size': self.noise_gaussian_size,
            'noise_gaussian_variability': self.noise_gaussian_variability,
            'reverse_streamlines_ratio': self.reverse_ratio,
            'split_streamlines_ratio': self.split_ratio,
            'step_size': self.step_size,
            'normalize_directions': self.normalize_directions
        }
        return hyperparameters

    @property
    def attributes(self):
        """
        All parameters. Contains at least all parameters that would be
        necessary to create this batch sampler again (except the dataset).
        """
        attrs = {
            'streamline_group_name': self.streamline_group_name,
            'batch_size': self.batch_size,
            'rng': self.rng,
            'nb_subjects_per_batch': self.nb_subjects_per_batch,
            'cycles': self.cycles,
            'step_size': self.step_size,
            'neighborhood_type': self.neighborhood_type,
            'neighborhood_radius': self.neighborhood_radius,
            'split_ratio': self.split_ratio,
            'noise_gaussian_size': self.noise_gaussian_size,
            'noise_gaussian_variability': self.noise_gaussian_variability,
            'reverse_ratio': self.reverse_ratio,
            'wait_for_gpu': self.wait_for_gpu,
            'normalize_directions': self.normalize_directions,
            'type': type(self)
        }
        return attrs

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
            self.dataset.total_streamlines[self.streamline_group])
        ids_per_subjs = \
            self.dataset.streamline_ids_per_subj[self.streamline_group]

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

            # if step_size, this is in mm. Else, it is is number of points.
            batch_size_per_subj = self.batch_size / nb_subjects

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
                    # Get the global streamline ids corresponding to this
                    # subject
                    subj_slice = ids_per_subjs[subj]

                    # We will continue iterating on this subject until we
                    # break (i.e. when we reach the maximum batch size for this
                    # subject)
                    total_subj_batch_size = 0
                    while True:
                        (subbatch_global_ids, subbatch_rel_ids,
                         subbatch_heaviness) = \
                            self._prepare_subj_subbatch(
                                subj, subj_slice, global_unused_streamlines,
                                total_subj_batch_size, batch_size_per_subj)

                        if subbatch_heaviness == 0:
                            # No streamlines remaining. Get next subject.
                            break

                        # Mask the sampled streamlines
                        global_unused_streamlines[subbatch_global_ids] = 0

                        # Add sample to batch
                        batch_ids_per_subj.append((subj, subbatch_rel_ids))

                        # Add this sub-batch's heaviness to total heaviness
                        total_subj_batch_size += subbatch_heaviness
                        if total_subj_batch_size > batch_size_per_subj:
                            # Batch size reached for this subject. Get next
                            # subject.
                            break
                self.log.debug("    Finished loop on subjects. Now yielding.\n"
                               "    (If this was called through a dataloader, "
                               "it should start using load_batch and even "
                               "training \n"
                               "     on this batch while we prepare a "
                               "batch for the next cycle, if any).")

                if len(batch_ids_per_subj) == 0:
                    self.log.debug("No more streamlines remain in any of the "
                                   "selected volumes! Restarting the batch "
                                   "sampler!")
                    break

                yield batch_ids_per_subj

            self.log.info("  Finished the cycles for these subjects. Choosing "
                          "new ones.")

    def _prepare_subj_subbatch(self, subj, subj_slice,
                               global_unused_streamlines, total_heaviness,
                               max_heaviness):

        # Find streamlines that have not been used yet for this subj
        subj_unused_ids_in_global = np.flatnonzero(
            global_unused_streamlines[subj_slice]) + subj_slice.start

        self.log.debug("    Available streamlines for this subject: {}"
                       .format(len(subj_unused_ids_in_global)))

        # No streamlines remain for this subject
        if len(subj_unused_ids_in_global) == 0:
            return [], [], 0

        # Sample a sub-batch of streamlines
        chosen_global_ids = self.np_rng.choice(subj_unused_ids_in_global,
                                                    CHUNK_SIZE)

        # Get their heaviness
        subj_data = self.dataset.subjs_data_list.open_handle_and_getitem(subj)
        subj_sft_data = subj_data.sft_data_list[self.streamline_group]
        if self.step_size:
            # batch_size is in mm.
            lengths = subj_sft_data.streamlines.lengths_mm[chosen_global_ids]
        else:
            # batch size in in number of points
            lengths = subj_sft_data.streamlines.lengths[chosen_global_ids]

        # If batch_size has been passed, taking a little less
        # streamlines for this last sub_batch.
        if total_heaviness + np.sum(lengths) >= max_heaviness:
            cumulative_sum = np.cumsum(lengths)
            selected = cumulative_sum < (max_heaviness - total_heaviness)
            chosen_global_ids = chosen_global_ids[selected]
            lengths = lengths[selected]

        sample_heaviness = np.sum(lengths)
        self.log.debug("    Chunk_size was {} but after verifying heaviness, "
                       "keeping only {} streamlines."
                       .format(CHUNK_SIZE, len(chosen_global_ids)))

        # Fetch subject-relative ids
        chosen_relative_ids = chosen_global_ids - subj_slice.start

        return chosen_global_ids, chosen_relative_ids, sample_heaviness,

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]]) \
            -> Union[Tuple[List, Dict], Tuple[List, List, List]]:
        """
        Fetches the chosen streamlines for all subjects in batch.
        Pocesses data augmentation.

        Torch uses this function to process the data with the dataloader
        parallel workers (on cpu). To be used as collate_fn.

        Parameters
        ----------
        streamline_ids_per_subj: dict[int, list]
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

            subj_data = self.dataset.subjs_data_list.open_handle_and_getitem(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group]

            # Get streamlines as sft
            sft = subj_sft_data.from_chosen_streamlines(s_ids)

            # Resampling streamlines to a fixed step size
            self.log.debug("            Resampling: {}".format(self.step_size))
            if self.step_size:
                # Note that we could skip resampling if it had already the same
                # step size, but computing current step size is not straight-
                # forward. Ex: A resampling of step_size = 1 may give a true
                # step_size of 0.99 (nb_points = int(length/step_size) creates
                # an approximation. We would need to define a tolerance on the
                # variability.
                sft = resample_streamlines_step_size(sft,
                                                     step_size=self.step_size)

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


class BatchStreamlinesSampler1IPV(BatchStreamlinesSampler):
    """
    This is used to load data under the form:

    x1: input = volume group named "input"
    x2: prev_dirs = a list of n previous dirs

    y: target = the whole streamlines as sequences.

    This is for instance the batch sampler used by Learn2Track and by
    Transformers.
    """

    def __init__(self, dataset: MultisubjectSubset,
                 streamline_group_name: str, input_group_name: str,
                 batch_size: int, rng: int, nb_subjects_per_batch: int,
                 cycles: int, step_size: float,
                 neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, Iterable[float], None],
                 split_ratio: float, noise_gaussian_size: float,
                 noise_gaussian_variability: float,
                 reverse_ratio: float, wait_for_gpu: bool,
                 normalize_directions: bool, nb_previous_dirs: int, **_):
        """
        Additional parameters compared to super:

        input_group_name: str
            Name of the volume group to load as input.
        nb_previous_dirs : int
            If set, concatenate the n previous streamline directions as input.
            [0].
        """
        super().__init__(dataset, streamline_group_name, batch_size, rng,
                         nb_subjects_per_batch, cycles, step_size,
                         neighborhood_type, neighborhood_radius,
                         split_ratio, noise_gaussian_size,
                         noise_gaussian_variability, reverse_ratio,
                         wait_for_gpu, normalize_directions)

        # This is probably the same as data_source.volume_groups, but asking
        # user again in case the dataset contain more than one input group
        # but you only want to use one.
        self.input_group_name = input_group_name

        # Find group index in the data_source
        idx = self.dataset.volume_groups.index(input_group_name)
        self.input_group_idx = idx

        if nb_previous_dirs is None:
            nb_previous_dirs = 0
        self.nb_previous_dirs = nb_previous_dirs
        self.final_nb_features = self._compute_nb_features(idx)

    def _compute_nb_features(self, input_group_idx):
        """
        Depending on data augmentation, compute features sizes to help prepare
        an eventual model.
        """
        # The nb_features is the last dim of each input volume.
        expected_input_size = self.dataset.nb_features[input_group_idx]

        if self.neighborhood_points is not None:
            expected_input_size += len(self.neighborhood_points) * \
                                   expected_input_size

        return expected_input_size

    @property
    def hyperparameters(self):
        hyperparameters = super().hyperparameters
        hyperparameters.update({'nb_previous_dirs': self.nb_previous_dirs})
        return hyperparameters

    @property
    def attributes(self):
        attributes = super().attributes
        attributes.update({'input_group_name': self.input_group_name,
                           'nb_previous_dirs': self.nb_previous_dirs})
        return attributes

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

            # If user chose to add neighborhood:
            if self.neighborhood_type:
                n_input_points = flat_subj_x_coords.shape[0]

                # Extend the coords array with the neighborhood coordinates
                flat_subj_x_coords = extend_coordinates_with_neighborhood(
                    flat_subj_x_coords, self.neighborhood_points)

                # Interpolate signal for each (new) point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float,
                                               device=device)
                flat_subj_x_data, coords_clipped = \
                    torch_trilinear_interpolation(data_volume, coords_torch)

                # Reshape signal into (n_points, new_nb_features)
                # DWI data features for each neighboor are contatenated.
                #    dwi_feat1_neig1  dwi_feat2_neig1 ...  dwi_featn_neighbm
                #  p1        .              .                    .
                #  p2        .              .                    .
                n_features = (flat_subj_x_data.shape[-1] *
                              (self.neighborhood_points.shape[0] + 1))
                subj_x_data = flat_subj_x_data.reshape(n_input_points,
                                                       n_features)
            else:  # No neighborhood:
                # Interpolate signal for each point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float,
                                               device=device)
                subj_x_data, coords_clipped = \
                    torch_trilinear_interpolation(data_volume.data,
                                                  coords_torch)

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
        """
        About device: see compute_inputs
        """
        # Compute previous directions
        previous_dirs = []
        if self.nb_previous_dirs > 0:
            empty_coord = torch.zeros((1, 3), dtype=torch.float32,
                                      device=device) * float('NaN')
            previous_dirs = \
                [torch.cat([torch.cat((empty_coord.repeat(i + 1, 1),
                                       s[:-(i + 1)]))
                            for i in range(self.nb_previous_dirs)], dim=1)
                 for s in batch_directions]
        return previous_dirs
