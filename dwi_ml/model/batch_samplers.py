# -*- coding: utf-8 -*-
"""
These batch samplers can be used in a torch DataLoader. For instance:
        # Initialize dataset
        dataset = MultiSubjectDataset(...)
        dataset.load_training_data()

        # Initialize batch sampler
        batch_sampler = BatchSampler(...)

        # Use this in the dataloader
        dataloader = DataLoader(dataset, batch_sampler=batch_sampler,
                                collate_fn=batch_sampler.load_batch)

The first class, BatchSamplerAbstract, defines functions that can be of use
for all models.

Depending on your model's needs in terms of streamlines, we offer to types of
batch samplers:
    - BatchSequenceSampler: provides functions in the case where streamlines
        are used as sequences, for instance in the Recurrent Neural Network or
        in a Transformer.
    - BatchPointSampler: provides functions in the case where streamlines are
        used locally, for instance in a Neural Nework or a Convolutional Neural
        Network.

You can then use these two batch samplers associated with the ones that fit
your other needs, for instance in terms of inputs. You are encouraged to
contribute to dwi_ml if no batch sampler fits your needs.
    - BatchSequenceSamplerOneInputVolume: In the simple case where you have one
        input per time step of the streamlines (ex, underlying dMRI
        information, or concatenated with other informations such as T1, FA,
        etc.). This is a child of the BatchSamplerSequence and is thus
        implemented to work with the whole streamlines as sequences.
        x = input
        y = sequences
"""

from collections import defaultdict
import logging
from typing import Dict, List, Union, Iterable, Iterator

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Sampler

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)
from dwi_ml.data.processing.space.neighbourhood import (
    extend_coordinates_with_neighborhood,
    get_neighborhood_vectors_axes, get_neighborhood_vectors_grid)
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)
from dwi_ml.data.processing.volume.interpolation import (
    torch_trilinear_interpolation)

# Number of streamlines sampled at the same for a given subject (then, we
# evaluate if the batch_size has been reached. No need to sample only one
# streamline at the time!
CHUNK_SIZE = 256


def prepare_neighborhood_information(neighborhood_type,
                                     neighborhood_radius):
    """
    Prepare neighborhood information for a given group.
    Always based on the first subject.
    Results are in the voxel world.
    """
    if neighborhood_type is not None:
        if neighborhood_type == 'axes':
            neighborhood_points = get_neighborhood_vectors_axes(
                neighborhood_radius)
        else:
            neighborhood_points = get_neighborhood_vectors_grid(
                neighborhood_radius)
        return neighborhood_points
    else:
        return None


class BatchSamplerAbstract(Sampler):
    """
    This class defines how to sample the streamlines available in the
    MultiSubjectData. However, the format in which the streamlines should be
    returned depend on how you want to use them: as whole sequences
    (torch.rnn.PackedSequence) or as individual time steps (NotImplemented).
    See below for the BatchSequenceSampler and the BatchPointSampler.

    It is possible to restrict the number of subjects in a batch (and thus the
    number of inputs to load associated with sampled streamlines), and to
    reduce the number of time we need to load new data by using the same
    subjects for a given number of "cycles".

    NOTE: Actual batch sizes might be different than `batch_size` depending
    on chosen data augmentation. This sampler takes streamline cutting and
    resampling into consideration, and as such, will return more (or less)
    points than the provided `batch_size`.

    Data augmentation is done on-the-fly to avoid having to multiply data on
    disk.
    """

    def __init__(self, data_source: Union[MultiSubjectDataset,
                                          LazyMultiSubjectDataset],
                 streamline_group_name: str,
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 step_size: float = None, neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_streamlines_ratio: float = 0.,
                 streamline_noise_sigma_mm: float = 0.,
                 reverse_streamlines_ratio: float = 0.5,
                 avoid_cpu_computations: bool = None,
                 device: torch.device = torch.device('cpu')):
        """
        Parameters
        ----------
        data_source : MultiSubjectDataset or LazyMultiSubjectDataset
            Dataset to sample from.
        streamline_group_name: str
            The name of the group to use to load the sequences. Probably
            'streamlines'. Should exist for all subjects in the
            MultiSubjectData.
        batch_size : int
            Number of required points in a batch. This will be approximated as
            the final batch size depends on data augmentation (streamline
            cutting or resampling). Note that approximation of the number of
            streamlines to fit this batch size will depend on the type of
            step_size: fixed or compressed data.
        rng : np.random.RandomState
            Random number generator.
        n_subject : int
            Optional; maximum number of subjects to be used in a single batch.
            Depending on the model, this can avoid loading too many input
            volumes at the same time, for example. If None, always sample
            from all subjects.
        cycles : int
            Optional, but required if `n_subjects` is given.
            Number of batches re-using the same volumes before sampling new
            ones.
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). If None, train on streamlines as they are (ex,
            compressed). Note that you probably already fixed a step size when
            creating your dataset, but you could use a different one here if
            you wish. [None]
        neighborhood_type: str, One of 'axes', 'grid' or None
            The type of neighborhood to add. If None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius_vox : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). [None]
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
        split_streamlines_ratio : float
            DATA AUGMENTATION: Percentage of streamlines to randomly split
            into 2, in each batch (keeping both segments as two independent
            streamlines).
            The reason for cutting is to help the ML algorithm to track from
            the middle of WM by having already seen half-streamlines. If you
            are using interface seeding, this is not necessary. Default: 0.
        streamline_noise_sigma_mm : float
            DATA AUGMENTATION: Add random Gaussian noise to streamline
            coordinates with given variance. Make sure it is smaller than your
            step size. Ex, you could choose 0.1 * step-size. Noise is truncated
            to +/- 2*noise_sigma. Default: 0.
            ToDo: add a variance in the distribution of noise between
             epoques. Comme ça, la même streamline pourra être vue plusieurs
             fois (dans plsr époques) mais plus ou moins bruitée d'une fois
             à l'autre.
        reverse_streamlines_ratio: float
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
        avoid_cpu_computations: bool
            If set, all computations that can be avoided in the batch sampler
            (which is computed on CPU) will be skipped and should be performed
            by the user later. Ex: computing directions from the streamline
            coordinates, computing input interpolation, etc.
        device: torch.device('cpu') or 'gpu'
            The device to use
        """
        super().__init__(data_source)  # This does nothing but python likes it.

        # Checking that batch_size is correct
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size <= 0):
            raise ValueError("batch_size should be a positive integeral "
                             "value, but got batch_size={}".format(batch_size))

        # Checking that n_volumes was given if cycles was given
        if cycles and not n_subject:
            raise ValueError("If `cycles_per_volume_batch` is defined, "
                             "`n_volumes` should be defined. Got: "
                             "n_volumes={}, cycles={}"
                             .format(n_subject, cycles))

        # Batch sampler variables
        self.data_source = data_source
        self.streamline_group_name = streamline_group_name
        self._rng = rng
        self.n_subjects = n_subject
        self.cycles = cycles
        self.step_size = step_size
        self.avoid_cpu_computations = avoid_cpu_computations
        self.device = device

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
        self.streamline_noise_sigma = streamline_noise_sigma_mm
        self.split_ratio = split_streamlines_ratio
        self.reverse_ratio = reverse_streamlines_ratio
        if not 0 <= self.split_ratio <= 1:
            raise ValueError('Split ratio must be a float between 0 and 1.')
        if not 0 <= self.reverse_ratio <= 1:
            raise ValueError('Reverse ration must be a float between 0 and 1')

        # Preparing the neighborhood for use by the child classes
        if neighborhood_type and not (neighborhood_type == 'axes' or
                                      neighborhood_type == 'grid'):
            raise ValueError("neighborhood type must be either 'axes', 'grid' "
                             "or None!")
        self.neighborhood_type = neighborhood_type
        if self.neighborhood_type is None and neighborhood_radius_vox:
            logging.warning("You have chosen not to add a neighborhood (value "
                            "None), but you have given a neighborhood radius. "
                            "Discarded.")
        self.neighborhood_points = prepare_neighborhood_information(
            neighborhood_type, neighborhood_radius_vox)

    def __iter__(self) -> Iterator[Dict[int, list]]:
        """
        Streamline sampling.

        First sample the subjects to be used from a given number of desired
        subjects, then sample streamline ids inside those volumes.

        Returns
        -------
        batch_ids_per_subj : dict[int, list]
            The list of streamline ids for each subject (relative ids inside
            each subject's tractogram).
        """
        # This is the list of all possible streamline ids
        global_streamlines_ids = np.arange(len(self.data_source))

        # This contains one bool per streamline:
        #   1 = this streamline has not been used yet.
        #   0 = this streamline has been used.
        global_available_streamlines = np.ones_like(global_streamlines_ids,
                                                    dtype=np.bool)

        # This will continue "yielding" batches until it encounters a break.
        # (i.e. when all streamlines have been used)
        while True:
            # Weight subjects by their number of remaining streamlines
            streamlines_per_subj = np.array(
                [np.sum(global_available_streamlines[subj_id_slice])
                 for _, subj_id_slice in
                 self.data_source.streamline_id_slice_per_subj.items()])
            logging.debug('Nb of remaining streamlines per subj: {}'
                          .format(streamlines_per_subj))
            assert (np.sum(streamlines_per_subj) ==
                    np.sum(global_available_streamlines)), \
                "Unexpected error, the total number of streamlines per " \
                "subject does not correspond to the total number of " \
                "streamlines in the multisubject dataset. Error in " \
                "streamline ids?"

            # Stopping if all streamlines have been used
            if np.sum(streamlines_per_subj) == 0:
                logging.info("No streamlines remain for this epoch, "
                             "stopping...")
                break

            # Choose subjects from which to sample streamlines for this batch
            if self.n_subjects:
                # Sampling first from subjects that were not seed a lot yet
                weights = streamlines_per_subj / np.sum(streamlines_per_subj)

                # Choosing only non-empty subjects
                n_subjects = min(self.n_subjects, np.count_nonzero(weights))
                sampled_subjs = self._rng.choice(
                    np.arange(len(self.data_source.data_list)),
                    size=n_subjects, replace=False, p=weights)
            else:
                # Sampling from all subjects
                sampled_subjs = self.data_source.streamline_id_slice_per_subj.keys()
                n_subjects = len(sampled_subjs)
            logging.debug('Sampled subjects for this batch: {}'
                          .format(sampled_subjs))

            # if step_size, this is in mm. Else, it is is number of points.
            batch_size_per_subj = self.batch_size / n_subjects

            # Preparing to iterate on these chosen subjects for a predefined
            # number of cycles
            if self.cycles:
                iterator = range(self.cycles)
            else:
                # Infinite iterator, sampling from all subjects
                iterator = iter(int, 1)
            for _ in iterator:
                batch_ids_per_subj = defaultdict(list)

                # For each subject, randomly choose streamlines that have not
                # been chosen yet.
                for subj in sampled_subjs:
                    # Get the global streamline ids corresponding to this
                    # subject
                    subj_id_slice = \
                        self.data_source.streamline_id_slice_per_subj[subj]

                    # We will continue iterating on this subject until we
                    # break (i.e. when we reach the maximum batch size for this
                    # subject)
                    total_subj_batch_size = 0
                    while True:
                        # Find streamlines that have not been used yet.
                        subj_available_ids = np.flatnonzero(
                            global_available_streamlines[subj_id_slice]) + \
                                             subj_id_slice.start
                        logging.debug("Available streamlines for this "
                                      "subject: {}"
                                      .format(len(subj_available_ids)))

                        # No streamlines remain for this subject
                        if len(subj_available_ids) == 0:
                            break

                        # Sample a sub-batch of streamlines and get their
                        # heaviness
                        sample_global_ids = \
                            self._rng.choice(subj_available_ids, CHUNK_SIZE)
                        subj_data = self.data_source.get_subject_data(subj)
                        if self.step_size:
                            # batch_size is in mm.
                            sample_heaviness = \
                                subj_data.sft_data.streamlines.lengths_mm[
                                    sample_global_ids]
                        else:
                            # batch size in in number of points
                            sample_heaviness = \
                                subj_data.sft_data.streamlines.lengths[
                                    sample_global_ids]

                        # If batch_size has been passed, taking a little less
                        # streamlines for this last sub_batch.
                        if (total_subj_batch_size + np.sum(sample_heaviness) >=
                                batch_size_per_subj):
                            cumulative_sum = np.cumsum(sample_heaviness)
                            selected = cumulative_sum < (batch_size_per_subj -
                                                         total_subj_batch_size)
                            sample_global_ids = sample_global_ids[selected]
                            sample_heaviness = sample_heaviness[selected]
                            volume_batch_fulfilled = True
                        else:
                            volume_batch_fulfilled = False

                        # Add this sub-batch's heaviness to total heaviness
                        total_subj_batch_size += np.sum(sample_heaviness)

                        # Mask the sampled streamline
                        global_available_streamlines[sample_global_ids] = 0

                        # Fetch subject-relative ids and add sample to batch
                        sample_relative_ids = \
                            sample_global_ids - subj_id_slice.start
                        batch_ids_per_subj[subj] = sample_relative_ids

                        if volume_batch_fulfilled:
                            break

                if len(batch_ids_per_subj) == 0:
                    logging.info("No more streamlines remain in any of the "
                                 "selected volumes! Moving to new cycle!")
                    break

                yield batch_ids_per_subj

    def _sft_data_augmentation(self, sft: StatefulTractogram):
        """ On-the-fly data augmentation."""

        # Adding noise to coordinates
        # Noise is considered in mm so we need to make sure the sft is in
        # rasmm space
        if self.streamline_noise_sigma and self.streamline_noise_sigma > 0:
            sft.to_rasmm()
            sft = add_noise_to_streamlines(sft, self.streamline_noise_sigma,
                                           self._rng)

        # Splitting streamlines
        # This increases the batch size, but does not change the total length
        if self.split_ratio and self.split_ratio > 0:
            all_ids = np.arange(len(sft))
            n_to_split = int(np.floor(len(sft) * self.split_ratio))
            split_ids = self._rng.choice(all_ids, size=n_to_split,
                                         replace=False)
            sft = split_streamlines(sft, self._rng, split_ids)

        # Reversing streamlines
        if self.reverse_ratio and self.reverse_ratio > 0:
            ids = np.arange(len(sft))
            self._rng.shuffle(ids)
            reverse_ids = ids[:int(len(ids) * self.reverse_ratio)]
            sft = reverse_streamlines(sft, reverse_ids)
        return sft

    def _get_volume_as_tensor(self, subj_idx: int, group_idx: int):
        """Here, get_subject_data is a LazySubjectData, its mri_data is a
        List[LazySubjectMRIData], not loaded yet but we will load it now using
        as_tensor.
        mri_group_idx corresponds to the group number from the config_file."""
        if self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                volume_data = self.volume_cache_manager[subj_idx]
            except KeyError:
                # volume_data isn't cached; fetch and cache it
                # This will open a hdf handle if it not created yet.
                mri = self.get_subject_data(subj_idx).mri_data_list[group_idx]
                volume_data = mri.as_tensor

                # Send volume_data on device and keep it there while it's
                # cached
                volume_data = volume_data.to(self.device)

                self.volume_cache_manager[subj_idx] = volume_data
            return volume_data
        else:
            # No cache is used
            mri = self.get_subject_data(subj_idx).mri_data_list[group_idx]
            return mri.as_tensor


class BatchSequencesSampler(BatchSamplerAbstract):
    """
    This class loads streamlines as a whole for sequence-based algorithms.
    Can be used as parent for your BatchSampler, depending on the type of
    data needed for your model, such as was done below with
    BatchSamplerOneInputVolumeSequence
    """

    def __init__(self, data_source: Union[MultiSubjectDataset,
                                          LazyMultiSubjectDataset],
                 streamline_group_name: str,
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 step_size: float = None, neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_streamlines_ratio: float = 0.,
                 streamline_noise_sigma_mm: float = 0.,
                 reverse_streamlines_ratio: float = 0.5,
                 avoid_cpu_computations: bool = None,
                 device: torch.device = torch.device('cpu'),
                 normalize_directions: bool = True):
        """
        Additional parameters compared to super:
        normalize_directions: bool
            If true, directions will be normalized. If the step size is fixed,
            it shouldn't make any difference. If streamlines are compressed,
            it theory you should normalize, but you could hope that not
            normalizing could give back to the algorithm a sense of distance
            between points.
        """
        super().__init__(data_source, streamline_group_name, batch_size, rng,
                         n_subject, cycles, step_size, neighborhood_type,
                         neighborhood_radius_vox, split_streamlines_ratio,
                         streamline_noise_sigma_mm, reverse_streamlines_ratio,
                         avoid_cpu_computations, device)
        self.normalize_directions = normalize_directions

    def load_batch(self, streamline_ids_per_subj: Dict[int, list]):
        """
        Fetches the chosen streamlines for all subjects in batch.
        Pocesses data augmentation.

        Torch uses this function to process the data with the dataloader
        workers. To be used as collate_fn. This part is ran on CPU.

        With self.avoid_cpu_computations options: directions are not computed.
        If you want to compute it later, use
        >> self.compute_and_normalize_directions(self, batch_streamlines)

        Parameters
        ----------
        streamline_ids_per_subj: dict[int, list]
            The list of streamline ids for each subject (relative ids inside
            each subject's tractogram).
        """
        batch_streamlines = []

        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        final_streamline_ids_per_subj = defaultdict(slice)
        for subj, s_ids in streamline_ids_per_subj.items():
            subj_data = self.data_source.get_subject_data(subj)
            logging.debug("    Data augmentation for subj {}"
                          .format(subj + 1))

            # Get streamlines as sft
            sft = subj_data.sft_data.get_chosen_streamlines_as_sft(s_ids)

            # Resampling streamlines to a fixed step size
            if self.step_size:
                # toDo Skip resampling if had already the same step size
                sft = resample_streamlines_step_size(sft,
                                                     step_size=self.step_size)

            # Data augmentation
            sft = self._sft_data_augmentation(sft)

            # Remember the indices of this subject's (augmented) streamlines
            ids_start = len(batch_streamlines)
            ids_end = ids_start + len(sft)
            final_streamline_ids_per_subj[subj] = slice(ids_start, ids_end)

            # Add all (augmented) streamlines to the batch
            batch_streamlines.extend(sft)

        if self.avoid_cpu_computations:
            return batch_streamlines, final_streamline_ids_per_subj
        else:
            packed_directions = \
                self.compute_and_normalize_directions(batch_streamlines)
            return (batch_streamlines, final_streamline_ids_per_subj,
                    packed_directions)

    def compute_and_normalize_directions(self, batch_streamlines):
        # Getting directions
        batch_directions = [torch.as_tensor(s[1:] - s[:-1],
                                            dtype=torch.float32,
                                            device=self.device)
                            for s in batch_streamlines]

        # Normalization:
        if self.normalize_directions:
            batch_directions = [s / torch.sqrt(torch.sum(s ** 2, dim=-1,
                                                         keepdim=True))
                                for s in batch_directions]
        packed_directions = pack_sequence(batch_directions,
                                          enforce_sorted=False)
        return packed_directions


class BatchPointsSampler(BatchSamplerAbstract):
    """
    This class loads streamlines one point at the time when streamlines
    are not needed as whole sequences.

    Currently, the __iter__ from the BatchSamplerAbstract samples complete
    streamline ids, not just points. The MultiSubjectData would have to be
    modified to find only some timesteps from a given streamline. It is
    probably easier to sample all points from each sampled streamline, and then
    use them separatedly instead of the rnn.PackedSequence that
    BatchSequenceSampler uses.
    """
    def __init__(self):
        raise NotImplementedError


class BatchSequencesSamplerOneInputVolume(BatchSequencesSampler):
    """
    This is used by torch with its collate_fn loading data as
    x: input = volume group named "input"
    y: target = the whole streamlines as sequences.

    This is for instance the batch sampler used by Learn2Track and by
    Transformers.
    """
    def __init__(self, data_source: Union[MultiSubjectDataset,
                                          LazyMultiSubjectDataset],
                 streamline_group_name: str, input_group_name: str,
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 step_size: float = None, neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_streamlines_ratio: float = 0.,
                 streamline_noise_sigma_mm: float = 0.,
                 reverse_streamlines_ratio: float = 0.5,
                 avoid_cpu_computations: bool = None,
                 device: torch.device = torch.device('cpu'),
                 normalize_directions: bool = True,
                 add_previous_dir: bool = False):
        """
        Additional parameters compared to super:

        input_group_name: str
            Name of the volume group to load as input.
        add_previous_dir : bool
            If set, concatenate the previous streamline direction as input.
            [False]
        """
        super().__init__(data_source, streamline_group_name, batch_size, rng,
                         n_subject, cycles, step_size, neighborhood_type,
                         neighborhood_radius_vox, split_streamlines_ratio,
                         streamline_noise_sigma_mm, reverse_streamlines_ratio,
                         avoid_cpu_computations, device, normalize_directions)

        self.input_group_name = input_group_name
        # Find group index in the data_source
        # Returns the first apperance. If the group is present twice, no error.
        # If the group is not present, will raise an error.
        idx = self.data_source.volume_groups.index(input_group_name)
        self.input_group_idx = idx

        self.add_previous_dir = add_previous_dir

    def load_batch(self, streamline_ids_per_subj: Dict[int, list]):
        """
        Fetches the chosen streamlines + underlying inputs for all subjects in
        batch. Pocesses data augmentation. Add previous_direction to the input.

        Torch uses this function to process the data with the dataloader
        workers. To be used as collate_fn. This part is ran on CPU.

        With self.avoid_cpu_computations options: directions are not computed,
        and interpolation is not done. If you want to compute them later, use
        >> self.compute_and_normalize_directions(self, batch_streamlines)
        >> self.compute_interpolation().

        Parameters
        ----------
        streamline_ids_per_subj: dict[int, list]
            The list of streamline ids for each subject (relative ids inside
            each subject's tractogram).

        Returns
        -------
        If self.do_interpolation is False:
            batch_streamlines : List of np.ndarray with shape (N_i,3)
                The streamlines coordinates in voxel space, ordered by subject.
            final_streamline_ids_per_subj : Dict[int, slice]
                A dictionary that maps each subject to the list of (processed)
                streamlines.
        else:
            packed_inputs : PackedSequence
                Inputs volume loaded from the given group name.
            packed_directions : PackedSequence
                Streamline directions.
        """

        if self.avoid_cpu_computations:
            # Only returning the streamlines for now.
            return super().load_batch(streamline_ids_per_subj)
        else:
            batch_streamlines, streamline_ids_per_subj, packed_directions = \
                super().load_batch(streamline_ids_per_subj)

            # Get the batch input volume
            # (i.e. volume's neighborhood under each point of the streamline)
            logging.debug("Testing something. This was using the unpacked "
                          "direction. I want to see if it is going to work "
                          "with a PackedSequence")
            batch_x = self.compute_interpolation(batch_streamlines,
                                                 streamline_ids_per_subj,
                                                 packed_directions)

            # Packing data.
            packed_inputs = pack_sequence(batch_x, enforce_sorted=False)

            return packed_inputs, packed_directions

    def compute_interpolation(self, batch_streamlines: List[np.ndarray],
                              streamline_ids_per_subj: Dict[int, slice],
                              batch_directions):
        """
        Get the DWI (depending on volume: as raw, SH, fODF, etc.) volume for
        each point in each streamline (+ depending on options: neighborhood,
        and preceding diretion)
        """
        batch_x_data = []
        for subj, y_ids in streamline_ids_per_subj.items():
            # Flatten = concatenate signal for all streamlines to process
            # faster. We don't use the last coord because it is used only to
            # compute the last target direction, it's not really an input
            flat_subj_x_coords = np.concatenate(
                [s[:-1] for s in batch_streamlines[y_ids]], axis=0)

            # Getting the subject's volume and sending to CPU/GPU
            data_volume = self.data_source.get_subject_mri_group_as_tensor(
                subj, self.input_group_idx, device=self.device,
                non_blocking=True)

            # If user chose to add neighborhood:
            if self.neighborhood_type:
                n_input_points = flat_subj_x_coords.shape[0]

                # Extend the coords array with the neighborhood coordinates
                flat_subj_x_coords = extend_coordinates_with_neighborhood(
                        flat_subj_x_coords, self.neighborhood_points)

                # Interpolate signal for each (new) point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float,
                                               device=self.device)
                flat_subj_x_data = torch_trilinear_interpolation(
                    data_volume, coords_torch)

                # Reshape signal into (n_points, new_feature_size)
                # DWI data features for each neighboor are contatenated.
                #    dwi_feat1_neig1  dwi_feat2_neig1 ...  dwi_featn_neighbm
                #  p1        .              .                    .
                #  p2        .              .                    .
                n_features = (flat_subj_x_data.shape[-1] *
                              self.neighborhood_points.shape[0])
                flat_subj_x_data = flat_subj_x_data.reshape(n_input_points,
                                                            n_features)
            else:  # No neighborhood:
                # Interpolate signal for each point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float,
                                               device=self.device)
                flat_subj_x_data = torch_trilinear_interpolation(
                    data_volume, coords_torch)

            # Free the data volume from memory "immediately"
            del data_volume

            # Split the flattened signal back to streamlines
            lengths = [len(s) - 1 for s in batch_streamlines[y_ids]]
            subbatch_x_data = flat_subj_x_data.split(lengths)
            batch_x_data.extend(subbatch_x_data)

        # Add previous directions to input
        if self.add_previous_dir:
            previous_dirs = [torch.cat((torch.zeros((1, 3),
                                                    dtype=torch.float32,
                                                    device=self.device),
                                        d[:-1]))
                             for d in batch_directions]
            batch_x_data = [torch.cat((s, p), dim=1)
                            for s, p in zip(batch_x_data, previous_dirs)]

        return batch_x_data
