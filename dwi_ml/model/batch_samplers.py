# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import logging
from typing import Dict, List, Tuple, Union, Iterable

import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Sampler

from dwi_ml.data.dataset.multi_subject_containers import (
    LazyMultiSubjectDataset, MultiSubjectDataset)
from dwi_ml.data.processing.space.neighbourhood import (
    get_neighborhood_vectors_axes, get_neighborhood_vectors_grid)
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)

"""
These batch samplers can then be used in a torch DataLoader. For instance:
        # Initialize dataset
        training_dataset = MultiSubjectDataset(...)

        # Initialize batch sampler
        training_batch_sampler = BatchSampler(...)

        # Use this in the dataloader
        training_dataloader = DataLoader(
            training_dataset, batch_sampler=BatchSampler,
            collate_fn=training_batch_sample.collate_fn)
            
The first class, BatchSamplerAbstract, defines functions that can be of use
for all models.

Depending on your model's needs in terms of streamlines, we offer to types of 
batch samplers:
    - BatchSamplerSequence: provides functions in the case where streamlines 
        are used as sequences, for instance in the Recurrent Neural Network or
        in a Transformer.
    - BatchSamplerPoint: provides functions in the case where streamlines are 
        used locally, for instance in a Neural Nework or a Convolutional Neural
        Network.
        
You can then use these two batch samplers associated with the ones that fit 
your other needs, for instance in terms of inputs. You are encouraged to 
contribute to dwi_ml if no batch sampler fits your needs.
    - BatchSamplerOneInputSequence: In the simple case where you have one input
    per time step of the streamlines (ex, underlying dMRI information, or 
    concatenated with other informations such as T1, FA, etc.). This is a child
    of the BatchSamplerSequence and is thus implemented to work with the whole
    streamlines as sequences.
        x = inputs
        y = sequences
"""
# Note that we use default noise variance for compressed streamlines,
# otherwise 0.1 * step-size
DEFAULT_NOISE_SIGMA_MM = 0.1
DEFAULT_REVERSE_RATIO = 0.5


class BatchSamplerAbstract(Sampler):
    """
    This class defines how to use data available in the MultiSubjectData:
       use noise? use the whole streamlines as sequences? use only one data
       point at the time? add neighboorhood? etc.

    Then it uses the sampled data to organize a batch. It is possible to
    restrict the number of volumes in a batch and to reduce the number of time
    we need to load new data, by using the same volumes for a given number of
    "cycles".

    NOTE: Actual batch sizes might be different than `batch_size` depending
    on chosen data augmentation. This sampler takes streamline cutting and
    resampling into consideration, and as such, will return more (or less)
    points than the provided `batch_size`.
    """

    def __init__(self, data_source: Union[MultiSubjectDataset,
                                          LazyMultiSubjectDataset],
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 use_streamline_noise: bool = False, step_size: float = None,
                 neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_ratio: float = None, add_previous_dir: bool = False):
        """
        Parameters
        ----------
        data_source : MultiSubjectDataset or LazyMultiSubjectDataset
            Dataset to sample from.
        batch_size : int
            Number of required points in a batch. This will be approximated as
            the final batch size depends on data augmentation (streamline
            cutting or resampling).
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
        use_streamline_noise : float
            If set, add random gaussian noise to streamline coordinates
            on-the-fly. Noise variance is 0.1 * step-size, or 0.1mm if no step
            size is used. [False]
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
        split_ratio : float
            Percentage of streamlines to randomly split into 2, in each batch.
            The reason for cutting is to help the ML algorithm to track from
            the middle of WM by having already seen half-streamlines. If you
            are using interface seeding, this is not necessary. If None, will
            not split streamlines. [None]
        add_previous_dir : bool
            If set, concatenate the previous streamline direction as input.
            [False]
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
        self.batch_size = batch_size
        self._rng = rng
        self.n_subjects = n_subject
        self.cycles = cycles

        self.step_size = step_size
        self.batch_size_length_mm = self.step_size * self.batch_size

        # Concerning the choice of dMRI data
        self.sh_order = None  # Stored in the dataset file

        # Concerning the choice of streamlines:
        # Noise, resampling, cutting, interpolation.
        self.do_interpolation = do_interpolation
        self.use_streamline_noise = use_streamline_noise
        self.split_ratio = split_ratio

        # Concerning the use of inputs
        self.add_previous_dir = add_previous_dir
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius_vox
        self.neighborhood_points = None  # Will be computed later
        if self.neighborhood_type is None:
            if self.neighborhood_radius:
                logging.warning('You have chosen not to add a neighborhood '
                                '(value None), but you have given a '
                                'neighborhood radius. Discarded.')
        if not (self.neighborhood_type == 'axes' or
                self.neighborhood_type == 'grid'):
            raise ValueError("neighborhood type must be either 'axes', 'grid' "
                             "or None!")

        if type(data_source) == LazyMultiSubjectDataset:
            self.hdf_file = data_source.hdf_handle

    def prepare_neighborhood_information(self):
        """
        Prepare neighborhood information for a given group.
        Always based on the first subject.
        Results are in the voxel world.
        """
        if self.neighborhood_type is not None:
            if self.neighborhood_type == 'axes':
                self.neighborhood_points = get_neighborhood_vectors_axes(
                    self.neighborhood_radius)
            else:
                self.neighborhood_points = get_neighborhood_vectors_grid(
                    self.neighborhood_radius)

    def __iter__(self):
        """First sample the volumes to be used from a given number of desired
        volumes, then sample streamline ids inside those volumes.

        Returns
        -------
        batch : list of tuple of (relative_streamline_id, tractodata_id)
        """
        # This is the list of all possible streamline ids
        global_streamlines_ids = np.arange(len(self.data_source))

        # This contains one bool per streamline:
        #   1 = this streamline has not been used yet.
        #   0 = this streamline has been used.
        global_streamlines_mask = np.ones_like(global_streamlines_ids,
                                               dtype=np.bool)

        # This will continue "yielding" batches until it encounters a break.
        # Possible breaks:
        #   - if all streamlines have been used
        #   -
        while True:
            # Weight subjects by their number of remaining streamlines
            streamlines_per_subj = np.array(
                [np.sum(global_streamlines_mask[start:end])
                 for subj_id, (start, end) in
                 self.data_source.subjID_to_streamlineID.items()])
            logging.debug('Nb of remaining streamlines per subj: {}'
                          .format(streamlines_per_subj))
            assert (np.sum(streamlines_per_subj) ==
                    np.sum(global_streamlines_mask)), \
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

                # Choose only non-empty subjects
                n_subjects = min(self.n_subjects, np.count_nonzero(weights))
                sampled_subjs = self._rng.choice(
                    np.arange(len(self.data_source.data_list)),
                    size=n_subjects, replace=False, p=weights)
            else:
                # sampling from all subjects
                sampled_subjs = self.data_source.subjID_to_streamlineID.keys()
                n_subjects = len(sampled_subjs)

            # Compute the number of *original* timesteps required per subject
            # (before resampling)
            # This is our indice of the heavyness of data.
            length_mm_per_subj = self.batch_size_length_mm / n_subjects

            # Preparing to iterate on these chosen subjects for a predefined
            # number of cycles
            if self.cycles:
                iterator = range(self.cycles)
            else:
                # Infinite iterator, sampling from all subjects
                iterator = iter(int, 1)

            for _ in iterator:
                # For each subject, randomly choose streamlines that have not
                # been chosen yet
                batch = []

                for subj in sampled_subjs:
                    # Get the global streamline ids corresponding to this
                    # subject
                    start, end = self.data_source.subjID_to_streamlineID[subj]
                    subj_global_ids = global_streamlines_ids[start:end]
                    logging.debug('Emma, batch sampler iter: '
                                  'Why is this different? There is '
                                  'something I have not understood?, should '
                                  'be the same. {}-{} == {}-{}? If so, just do'
                                  'np.array(start:end)'
                                  .format(start, end, subj_global_ids[0],
                                          subj_global_ids[-1]))

                    # We will continue iterating on this subject until we
                    # reach the maximum lengths_mm for this subject
                    total_subj_length_mm = 0
                    while True:
                        # Filter for available (unmasked) streamlines
                        available_streamline_ids = \
                            subj_global_ids[
                                global_streamlines_mask[start:end]]

                        # No streamlines remain for this volume
                        if len(available_streamline_ids) == 0:
                            break

                        # Sample a batch of streamlines and get their lengths
                        sample_global_ids = \
                            self._rng.choice(available_streamline_ids, 256)
                        sample_lengths_mm = \
                            self.data_source.streamline_lengths_mm[
                                sample_global_ids]

                        volume_batch_fulfilled = False
                        # Keep total volume length under the maximum
                        if (total_subj_length_mm + np.sum(
                                sample_lengths_mm) >
                                length_mm_per_subj):
                            # Select only enough streamlines to fill the
                            # required length
                            cumulative_sum = np.cumsum(sample_lengths_mm)
                            selected_mask = cumulative_sum < (
                                    length_mm_per_subj - total_subj_length_mm)
                            sample_global_ids = sample_global_ids[
                                selected_mask]
                            sample_lengths_mm = sample_lengths_mm[
                                selected_mask]
                            volume_batch_fulfilled = True

                        # Add this streamline's length to total length
                        total_subj_length_mm += np.sum(sample_lengths_mm)

                        # Mask the sampled streamline
                        global_streamlines_mask[sample_global_ids] = 0

                        # Fetch tractodata relative id
                        sample_relative_ids = sample_global_ids - start

                        # Add sample to batch
                        for sample_id in sample_relative_ids:
                            batch.append((sample_id, subj))

                        if volume_batch_fulfilled:
                            break

                if len(batch) == 0:
                    logging.info("No more streamlines remain in any of the "
                                 "selected volumes! Moving to new cycle!")
                    break

                yield batch


class BatchSequencesSampler(BatchSamplerAbstract):
    """
    This class loads streamlines as a whole for sequence-based algorithms.
    Can be used as parent for your BatchSampler, depending on the type of
    data needed for your model, such as was done below with
    BatchSamplerOneInputVolumeSequence

    Data augmentation
    -----------------
    The _get_batch_streamlines method does on-the-fly data augmentation, to
    avoid doubling the data on disk:
    - Resampling step size
    - Adding Gaussian noise (truncated to +/- 2*noise_sigma)
        ToDo: add a variance in the distribution of noise between
         epoques. Comme ça, la même streamline pourra être vue plusieurs
         fois (dans plsr époques) mais plus ou moins bruitée d'une fois
         à l'autre.
    - Cutting streamlines: Splitting some streamlines into 2 at random
    positions and keeping both segments as two independent streamlines
        .The number of streamlines to split depends on the split_ratio.
        .This increases the batch size, but does not change the number of
        timesteps.
        .We need to do it subject per subject to keep track of the streamline
         ids.
    - Reversing half of streamline batch: You could want to reverse ALL your
    data and then use both the initial data and reversed data. But this would
    take twice the memory. Here, for each epoch, you have 50% chance to be
    reversed. If you train for enough epochs, high chance that you will have
    used both directions of your streamline at least once.
        A way to absolutely ensure using both directions the same number of
        time, we could use a flag and at each epoch, reverse those with
        unreversed flag. But that adds a bool for each streamline in your
        dataset and probably not so useful.
    """

    def __init__(self, streamline_group_name: str,
                 data_source: Union[MultiSubjectDataset,
                                    LazyMultiSubjectDataset],
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 use_streamline_noise: bool = False, step_size: float = None,
                 neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_ratio: float = None, add_previous_dir: bool = False,
                 compute_directions: bool = False,
                 normalize_directions: bool = True):
        """
        streamline_group_name: str
            The name of the group to use for the sequences. Probably
            'streamlines'. Should exist for all subjects.

        normalize_directions: bool
            If true, directions will be normalized. If the step size is fixed,
            it shouldn't make any difference. If streamlines are compressed,
            it theory you should normalize, but you could hope that not
            normalizing could give back to the algorithm a sense of distance
            between points.
        """
        super().__init__(data_source, batch_size, rng, n_subject, cycles,
                         use_streamline_noise, step_size,
                         neighborhood_type,
                         neighborhood_radius_vox, split_ratio,
                         add_previous_dir)

        self.streamline_group_name = streamline_group_name
        self.normalize_directions = normalize_directions

    @staticmethod
    def _get_batch_streamline_ids_per_subj(batch_streamline_ids: List[int]):
        """
        batch_ids: List[int]
            The list of streamlines to process for this batch.

        Returns the list of same ids, reordered per subject to accelerate
        processing.
        """
        batch_streamline_ids_per_subj = defaultdict(list)
        for y_id, subj in batch_streamline_ids:
            batch_streamline_ids_per_subj[subj].append(y_id)

        logging.debug("The streamlines ids for this batch will be: {}"
                      .format(dict(batch_streamline_ids_per_subj)))
        return batch_streamline_ids_per_subj

    def _get_batch_streamlines(self, streamline_ids_per_subj):
        """
        Fetches the list of streamlines for all subjects in batch + processes
        data augmentation.

        Torch uses this function to process the data with the dataloader
        workers. To be used as collate_fn. This part is ran on CPU.

        We can choose which parts t

        FUNCTIONNING:
            - With interpolation: Gets the list of :
                X = the dwi data under each point of the streamlines
                streamlines = the streamlines coordinates.
                directions = the streamlines directions.
            Should be used with big data.
            - Without interpolation: Only gets the list of streamlines
            coordinates. The rest (X, directions) will be done by the training
            script directly on GPU.
            Can be used with smaller data.

        Parameters
        ----------
        streamline_ids_per_subj: Dict with the list of streamlines to get for
            each subject.
        """
        batch_streamlines = []

        # The batch's streamline ids will change throughout processing because
        # of data augmentation. These final ids will correspond to loaded,
        # processed streamlines, not to the ids in the hdf5 file.
        final_streamline_ids_per_subj = OrderedDict()

        for subj, s_ids in streamline_ids_per_subj.items():
            subj_data = self.data_source.get_subject_data(subj)
            logging.debug("    Data augmentation for subj {}"
                          .format(subj + 1))

            # Get streamlines as sft
            sub_sft = subj_data.sft_data.get_chosen_streamlines_as_sft(s_ids)

            # On-the-fly data augmentation:

            # Resampling streamlines to a fixed step size
            if self.step_size:
                sub_sft = resample_streamlines_step_size(
                    sub_sft, step_size=self.step_size)

            # Adding noise to coordinates
            # Noise is considered in mm so we need to make sure the sft is in
            # rasmm space
            if self.use_streamline_noise:
                sub_sft.to_rasmm()
                noise_mm = DEFAULT_NOISE_SIGMA_MM * (self.step_size or 1.)
                sub_sft = add_noise_to_streamlines(sub_sft, noise_mm,
                                                   self._rng)

            # Splitting streamlines
            if self.split_ratio:
                all_ids = np.arange(len(sub_sft))
                n_to_split = int(np.floor(len(sub_sft) * self.split_ratio))
                split_ids = self._rng.choice(all_ids, size=n_to_split,
                                             replace=False)
                sub_sft = split_streamlines(sub_sft, self._rng, split_ids)

            # Reversing streamlines
            ids = np.arange(len(sub_sft))
            self._rng.shuffle(ids)
            reverse_ids = ids[:int(len(ids) * DEFAULT_REVERSE_RATIO)]
            sub_sft = reverse_streamlines(sub_sft, reverse_ids)

            # Remember the indices of this subject's (augmented) streamlines
            ids_start = len(batch_streamlines)
            ids_end = ids_start + len(sub_sft)
            final_streamline_ids_per_subj[subj] = slice(ids_start, ids_end)

            # Add all (augmented) streamlines to the batch
            batch_streamlines.extend(sub_sft)

        if self.do_interpolation:
            # Getting directions
            batch_directions = self._get_batch_directions(batch_streamlines)

            # Normalization:
            if self.normalize_directions:
                batch_directions = [s / torch.sqrt(torch.sum(s ** 2, dim=-1,
                                                             keepdim=True))
                                    for s in batch_directions]
            packed_directions = pack_sequence(batch_directions,
                                              enforce_sorted=False)
            return packed_directions
        else:
            return batch_streamlines, final_streamline_ids_per_subj

    def _get_batch_directions(self, batch_streamlines: List[np.ndarray]):
        """
        Get the direction between two adjacent points for each streamlines.
        """
        directions = [torch.as_tensor(s[1:] - s[:-1], dtype=torch.float32,
                                      device=self.data_source.device)
                      for s in batch_streamlines]
        return directions


class BatchPointsSampler(BatchSamplerAbstract):
    """
    This class loads streamlines one point at the time when streamlines
    are not needed as whole sequences.
    """
    raise NotImplementedError


class BatchSequencesSamplerOneInputVolume(BatchSequencesSampler):
    """
    This is used by torch with its collate_fn loading data as
    x: input = volume group named "input"
    y: target = the whole streamlines as sequences.

    This is for instance the batch sampler used by Learn2Track and by
    Transformers.
    """

    def __init__(self, input_group_name: str, streamline_group_name: str,
                 data_source: Union[MultiSubjectDataset,
                                    LazyMultiSubjectDataset],
                 batch_size: int, rng: np.random.RandomState,
                 n_subject: int = None, cycles: int = None,
                 use_streamline_noise: bool = False, step_size: float = None,
                 neighborhood_type: str = None,
                 neighborhood_radius_vox: Union[int, float,
                                                Iterable[float]] = None,
                 split_ratio: float = None, add_previous_dir: bool = False,
                 do_interpolation: bool = False,
                 normalize_directions: bool = True):
        """
        Additional parameters compared to super:

        input_group_name: str
            Name of the volume group to load as input.
        """
        super().__init__(streamline_group_name, data_source,
                         batch_size, rng, n_subject, cycles,
                         use_streamline_noise, step_size, neighborhood_type,
                         neighborhood_radius_vox, split_ratio,
                         add_previous_dir, do_interpolation)

        self.input_group_name = input_group_name
        # Find group index in the data_source
        # Returns the first apperance. If the group is present twice, no error.
        # If the group is not present, will raise an error.
        idx = self.data_source.volume_groups.index(input_group_name)
        self.input_group_idx = idx

    def get_batch(self, batch_ids: List[Tuple[int, int]]):
        """
        PURPOSE: Torch uses this function to process the data with the
        dataloader workers. To be used as collate_fn.

        CPU: It means that this part is ran on CPU. Particularly interesting
        for interpolation: we don't need to send the DWI data to GPU.
        Interpolation is done on CPU and interpolated data is send to GPU for
        the model_and_training.

        FUNCTIONNING:
            - With interpolation: Gets the list of :
                X = the dwi data under each point of the streamlines
                streamlines = the streamlines coordinates.
                directions = the streamlines directions.
            Should be used with big data.
            - Without interpolation: Only gets the list of streamlines
            coordinates. The rest (X, directions) will be done by the training
            script directly on GPU.
            Can be used with smaller data.

        Parameters
        ----------
        batch_ids : List of tuple of (int,int)
            A list of (streamline_id, subject_id), the streamlines to get for
            this batch.

        Returns
        -------
        If self.do_interpolation is False:
            batch_streamlines : List of np.ndarray with shape (N_i,3)
                The streamlines coordinates in voxel space, ordered by subject.
            tid_to_subbactch_sid : dict of [int,slice]
                A dictionary that maps each tractodata_id to a subbatch of
                voxel_streamlines (i.e. a slice).
        else:
            packed_inputs : PackedSequence
                Inputs volume loaded from the given group name.
            packed_directions : PackedSequence
                Streamline directions.
        """

        # Get the batch augmented streamlines and their new ids.
        batch_streamlines, streamline_ids_per_subj = \
            self._get_batch_streamlines(batch_ids)

        if self.do_interpolation:
            # Get the batch input volume and the streamline directions
            packed_directions = self._get_batch_streamlines(batch_ids)

            # Get the batch input volume
            # (i.e. volume's neighborhood under each point of the streamline)
            batch_x = self._get_batch_x_interp(batch_streamlines,
                                               streamline_ids_per_subj,
                                               batch_directions)

            # Packing data.
            packed_inputs = pack_sequence(batch_x, enforce_sorted=False)

            return packed_inputs, packed_directions
        else:
            # Only returning the streamlines for now.
            return batch_streamlines, streamline_ids_per_subj

    def _get_batch_x_interp(self, batch_streamlines: List[np.ndarray],
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
            # compute the last target direction from y, it's not really an
            # input
            flat_subj_x_coords = np.concatenate(
                [s[:-1] for s in batch_streamlines[y_ids]], axis=0)

            # Getting the subject's volume and sending to CPU/GPU
            data_volume = self.get_subject_mri_group_as_tensor(subj)
            data_volume = data_volume.to(device=device, non_blocking=True)

            # If user chose to add neighborhood:
            if self.add_neighborhood_mm:
                n_input_points = flat_subj_x_coords.shape[0]

                # Extend the coords array with the neighborhood coordinates
                flat_subj_x_coords = \
                    extend_coords_with_interp_neighborhood_vectors(
                        flat_subj_x_coords, self.neighborhood_vectors)

                # Interpolate signal for each (new) point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float,
                                               device=device)
                flat_subj_x_data = torch_trilinear_interpolation(
                    data_volume, coords_torch)

                # Reshape signal into (n_points, new_feature_size)
                # toDo DWI data features for each neighboor are contatenated.
                #    dwi_feat1_neig1  dwi_feat2_neig1 ...  dwi_featn_neighbm
                #  p1        .              .                    .
                #  p2        .              .                    .
                #  Won't work for CNN!?
                n_features = (flat_subj_x_data.shape[-1] *
                              self.neighborhood_vectors.shape[0])
                flat_subj_x_data = flat_subj_x_data.reshape(n_input_points,
                                                            n_features)
            else:  # No neighborhood:
                # Interpolate signal for each point
                coords_torch = torch.as_tensor(flat_subj_x_coords,
                                               dtype=torch.float)
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
                                                    device=device),
                                        d[:-1]))
                             for d in batch_directions]
            batch_x_data = [torch.cat((s, p), dim=1)
                            for s, p in zip(batch_x_data, previous_dirs)]

        return batch_x_data
