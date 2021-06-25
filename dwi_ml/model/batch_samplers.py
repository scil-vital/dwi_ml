# -*- coding: utf-8 -*-
from collections import OrderedDict, defaultdict
import logging
import os
from typing import Dict, List, Tuple

from nibabel.streamlines import ArraySequence
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset, Sampler

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset

# Note that we use default noise variance for compressed streamlines,
# otherwise 0.1 * step-size
DEFAULT_NOISE_MM = 0.1


# toDo : Needs cleaning.
#  See here : VITALabAI/project/learn2track/scripts/training.py
class BatchSampler(Sampler):
    """Samples sequences using a number of required timesteps, without
    replacement.

    It is also possible to restrict of volumes in a batch, in which case
    a number of "cycles" is also required, which for how many batches the
    same volumes should be re-used before sampling new ones.

    NOTE: Actual batch sizes might be different than `batch_size` depending
    on later data augmentation. This sampler takes streamline cutting and
    resampling into consideration, and as such, will return more (or less)
    points than the provided `batch_size`.

    Arguments:
        data_source : Dataset
            Dataset to sample from.
        batch_size : int
            Number of required points in a batch. This will be approximated as
            the final batch size depends on data augmentation (streamline cutting
            or resampling).
        rng : np.random.RandomState
            Random number generator.
        n_volumes : int
            Optional; maximum number of volumes to be used in a single batch.
            If None, always use all volumes.
        cycles : int
            Optional, but required if `n_volumes` is given.
            Number of batches re-using the same volumes before sampling new ones.
    """

    def __init__(self, data_source: MultiSubjectDataset,  # Says error because it doesn't use super().__init__
                 batch_size: int, rng: np.random.RandomState,
                 n_volumes: int = None, cycles: int = None):
        if not isinstance(data_source, Dataset):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Dataset, but got data_source={}"
                             .format(data_source))
        if (not isinstance(batch_size, int) or isinstance(batch_size, bool)
                or batch_size <= 0):
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if cycles and not n_volumes:
            raise ValueError("If `cycles_per_volume_batch` is defined, "
                             "`n_volumes` should be defined. Got: "
                             "n_volumes={}, cycles={}"
                             .format(n_volumes, cycles))
        self.data_source = data_source
        self.batch_size = batch_size
        self._rng = rng
        self.n_volumes = n_volumes
        self.cycles = cycles

        self.batch_size_length_mm = self.data_source.step_size * self.batch_size

    def __iter__(self):
        """First sample the volumes to be used from a given number of desired
        volumes, then sample streamline ids inside those volumes.

        Returns
        -------
        batch : list of tuple of (relative_streamline_id, tractodata_id)
        """
        global_streamlines_ids = np.arange(len(self.data_source))
        global_streamlines_mask = np.ones_like(global_streamlines_ids,
                                               dtype=np.bool)

        while True:
            # Weight volumes by their number of remaining streamlines
            streamlines_per_volume = np.array(
                [np.sum(global_streamlines_mask[start:end])
                 for tid, (start, end) in
                 self.data_source.subjID_to_streamlineID.items()])

            if np.sum(streamlines_per_volume) == 0:
                logging.info("No streamlines remain for this epoch, "
                             "stopping...")
                break

            if self.n_volumes:
                weights = \
                    streamlines_per_volume / np.sum(streamlines_per_volume)

                # Choose only non-empty volumes
                n_volumes = min(self.n_volumes, np.count_nonzero(weights))
                sampled_tids = self._rng.choice(
                    np.arange(len(self.data_source.data_list)),
                    size=n_volumes, replace=False, p=weights)
            else:
                sampled_tids = self.data_source.subjID_to_streamlineID.keys()
                n_volumes = len(sampled_tids)

            # Compute the number of *original* timesteps required per volume
            # (before resampling)
            length_mm_per_volume = self.batch_size_length_mm / n_volumes

            if self.cycles:
                iterator = range(self.cycles)
            else:
                # Infinite iterator
                iterator = iter(int, 1)

            for _ in iterator:
                # For each volume, randomly choose streamlines that haven't been
                # chosen yet
                batch = []

                for tid in sampled_tids:
                    # Get the global streamline ids corresponding to this volume
                    start, end = self.data_source.subjID_to_streamlineID[tid]
                    volume_global_ids = global_streamlines_ids[start:end]

                    total_volume_length_mm = 0
                    while True:
                        # Filter for available (unmasked) streamlines
                        available_streamline_ids = \
                            volume_global_ids[global_streamlines_mask[start:end]]

                        # No streamlines remain for this volume
                        if len(available_streamline_ids) == 0:
                            break

                        # Sample a batch of streamlines and get their lengths
                        sample_global_ids = \
                            self._rng.choice(available_streamline_ids, 256)
                        sample_lengths_mm = \
                            self.data_source.streamline_lengths_mm[sample_global_ids]

                        volume_batch_fulfilled = False
                        # Keep total volume length under the maximum
                        if (total_volume_length_mm + np.sum(sample_lengths_mm) >
                            length_mm_per_volume):
                            # Select only enough streamlines to fill the
                            # required length
                            cumulative_sum = np.cumsum(sample_lengths_mm)
                            selected_mask = cumulative_sum < (length_mm_per_volume - total_volume_length_mm)
                            sample_global_ids = sample_global_ids[
                                selected_mask]
                            sample_lengths_mm = sample_lengths_mm[
                                selected_mask]
                            volume_batch_fulfilled = True

                        # Add this streamline's length to total length
                        total_volume_length_mm += np.sum(sample_lengths_mm)

                        # Mask the sampled streamline
                        global_streamlines_mask[sample_global_ids] = 0

                        # Fetch tractodata relative id
                        sample_relative_ids = sample_global_ids - start

                        # Add sample to batch
                        for sample_id in sample_relative_ids:
                            batch.append((sample_id, tid))

                        if volume_batch_fulfilled:
                            break

                if len(batch) == 0:
                    logging.info("No more streamlines remain in any of the "
                                 "selected volumes! Moving to new cycle!")
                    break

                yield batch


class TrainingBatchSamplerOneInputVolume(MultiSubjectDataset):
    """
    This is used by torch with its collate_fn loading data as
    x: input = volume group named "input"
    y: target = the streamlines

    This is for instance the batch sampler used by Learn2Track and by
    Transformers.
    """
    def __init__(self, hdf5_path: str, rng: np.random.RandomState,
                 name: str = None, use_streamline_noise: bool = False,
                 step_size: float = None,
                 neighborhood_dist_mm: float = None, nb_neighborhood_axes=6,
                 streamlines_cut_ratio: float = None,
                 add_previous_dir: bool = False,
                 do_interpolation: bool = False,
                 device: torch.device = torch.device('cpu'),
                 taskman_managed: bool = False):
        """
        Parameters
        ----------
        hdf5_path : str
            Path to the processed .hdf5 file.
        rng : np.random.RandomState
            Random number generator.
        name : str
            Name of the dataset. If none, use the basename of the given .hdf5
            file. [None]
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
        neighborhood_dist_mm : float
            Add neighborhood points at the given distance (in mm) in each
            direction (nb_neighborhood_axes). [None] (None and 0 have the same
            effect).
        nb_neighborhood_axes: int
            Nb of axes (directions) to get the neighborhood voxels. This is
            only used if do_interpolation is True. Currently, 6 is the default
            and only implemented version (left, right, front, behind, up,
            down).
        streamlines_cut_ratio : float
            Percentage of streamlines to randomly cut in each batch. The reason
            for cutting is to help the ML algorithm to track from the middle of
            WM by having already seen half-streamlines. If you are using
            interface seeding, this is not necessary. If None, do not split
            streamlines. [None]
        add_previous_dir : bool
            If set, concatenate the previous streamline direction as input.
            [False]
        do_interpolation : bool
            If True, do the interpolation in the collate_fn (worker function).
            In this case, collate_fn returns PackedSequences ready for the
            model. [False]
        device : torch.device
            Device on which to process data. ['cpu']
        taskman_managed : bool
            If True, taskman manages the experiment. Do not output progress bars
            and instead output special messages for taskman. [False]
        """
        super().__init__(hdf5_path, name, device, taskman_managed)

        # Concerning the choice of streamlines:
        # Noise, resampling, cutting, interpolation.
        self.do_interpolation = do_interpolation
        self.default_noise_mm = DEFAULT_NOISE_MM
        self.use_streamline_noise = use_streamline_noise
        self.step_size = step_size
        self.streamlines_cut_ratio = streamlines_cut_ratio

        # Concerning the use of inputs
        self.add_previous_dir = add_previous_dir
        self.add_neighborhood_mm = neighborhood_dist_mm
        self.neighborhood_vectors = None  # Will be computed later
        self.nb_neighborhood_axes = nb_neighborhood_axes

        # Concerning the choice of dMRI data
        self.sh_order = None  # Stored in the dataset file
        
        # Concerning the memory usage:
        self.rng = rng

    def prepare_neighborhood_information(self):
        # Get neighborhood information.
        if self.add_neighborhood_mm:
            #non-lazy:
            affine = self.data_list[0].mri_data_list.affine

            #lazy:
            affine = self.tractodata_manager[(0, hdf_file)].input_dv.affine

            self.neighborhood_vectors = get_interp_neighborhood_vectors(
                self.add_neighborhood_mm, nb_axes=self.nb_neighborhood_axes,
                convert_mm_to_vox=True, affine=affine)

    # Functions, methods and properties already defined in super:
    # Everything necessary to load data from the hdf5 file.
    #   load_data
    #   arrange streamline
    #   get_subject_data
    #   get_subject_mri_data
    #   get_subject_streamlines_subset
    #   create_subject_from_hdf
    #   __len__
    #   __getitem__

    def get_batch_x_y_and_target(self, batch_ids: List[Tuple[int, int]]):
        """ COLLATE_FN.

        PURPOSE: Torch uses this function to process the data with the
        dataloader workers.

        CPU: It means that this part is ran on CPU. Particularly interesting
        for interpolation: we don't need to send the DWI data to GPU.
        Interpolation is done on CPU and interpolated data is send to GPU for
        the model_and_training.

        FUNCTIONNING:
            - With interpolation: Gets the list of (X, target) examples for the
            model_and_training/validation set, where
                X = the dwi data under each point of the streamlines
                Y = the streamlines coordinates.
                Target = the "ground truth" direction between each point Y.
            Should be used with big data.
            - Without interpolation: Only gets the list of Y streamlines
            coordinates. The rest will be done by the model_and_training script
            directly on GPU.
            Can be used with smaller data.

        Parameters
        ----------
        batch_ids : List of tuple of (int,int)
            A list of (streamline_id, subject_id).

        Returns
        -------
        If self.do_interpolation is False:
            voxel_streamlines : List of np.ndarray with shape (N_i,3)
                The streamlines coordinates in voxel space.
                The streamlines are ordered by tractodata.
            tid_to_subbactch_sid : dict of [int,slice]
                A dictionary that maps each tractodata_id to a subbatch of
                voxel_streamlines (i.e. a slice).
        else:
            packed_inputs : PackedSequence
                Inputs for the model
            packed_targets : PackedSequence
                Targets for the model
        """

        # streamline_ids are unordered, so first group them by subject
        batch_subj_to_y_id = defaultdict(list)
        for y_id, subj in batch_ids:
            batch_subj_to_y_id[subj].append(y_id)

        # Get the batch y (streamlines)
        batch_streamlines, batch_subj_to_y_ids_processed = \
            self._get_batch_y(batch_subj_to_y_id)

        if self.do_interpolation:
            # Get the batch target (ground truth directions)
            batch_directions = self._get_batch_target(batch_streamlines,
                                                      device=self.device)

            # Get the batch X (dWI volume under each point of the streamline)
            # NOTE. If we add previous_direction to input X, we don't normalize
            # it. In the case of compressed streamlines, we hope this will give
            # back to the algorithm a sense of distance between point.
            batch_x = self._get_batch_x_interp(batch_streamlines,
                                               batch_subj_to_y_ids_processed,
                                               batch_directions,
                                               device=self.device)

            # Now, normalize targets. If the step size is always the same,
            # shouldn't make any difference. If compressed.... discutable
            # choice.
            #                                                                                               toDo à discuter en groupe
            targets = [s / torch.sqrt(torch.sum(s ** 2, dim=-1, keepdim=True))
                       for s in batch_directions]

            # Packing data.
            # `enforce_sorted=False` will sort sequences automatically before
            # packing.
            packed_inputs = pack_sequence(batch_x, enforce_sorted=False)
            packed_targets = pack_sequence(targets, enforce_sorted=False)

            return packed_inputs, packed_targets
        else:
            return batch_streamlines, batch_subj_to_y_ids_processed

    def _get_batch_y(self, batch_subj_to_y_id):
        """
        Fetch the list of y (streamlines) for all subjects in batch.

        Note. The streamlines are modified here (resampling step size, adding
        noise, cutting streamlines, flipping streamlines)
        """
        batch_streamlines = []
        batch_subj_to_y_ids_processed = OrderedDict()
        for subj, y_ids in batch_subj_to_y_id.items():
            # Get streamlines
            subject_streamlines = self.get_subject_streamlines_subset(subj,
                                                                      y_ids)

            # Make sure streamlines are stored in a list
            if isinstance(subject_streamlines, ArraySequence):
                subject_streamlines = [s for s in subject_streamlines]

            # Get affine. Used to preprocess streamlines
            affine_vox2rasmm = \
                self.get_subject_data(subj).dmri_data.affine_vox2rasmm

            # Resample streamlines to a fixed step size
            if self.step_size:
                subject_streamlines = resample_streamlines_step_size(
                    subject_streamlines, step_size=self.step_size)

            # Add noise to coordinates
            # ToDo: add a variance in the distribution of noise between epoques.
            #  Comme ça, la même streamline pourra être vue plusieurs fois
            #  (dans plsr époques) mais plus ou moins bruitée d'une fois à
            #  l'autre.
            if self.use_streamline_noise:
                noise_mm = self.default_noise_mm * (self.step_size or 1.)
                subject_streamlines = add_noise_to_streamlines(
                    subject_streamlines, noise_mm, self.rng,
                    convert_mm_to_vox=True, affine=affine_vox2rasmm)

            # Cut streamlines in random positions, using the same ratio for each
            # Cutting keeps both segments as two independent streamlines, and
            # increases the batch size (but does not change the number of
            # timesteps). Need to do it subject per subject to keep track of the
            # streamline ids.
            if self.streamlines_cut_ratio:
                all_ids = np.arange(len(subject_streamlines))
                n_to_split = int(np.floor(len(subject_streamlines) *
                                          self.streamlines_cut_ratio))
                split_ids = self.rng.choice(all_ids, size=n_to_split,
                                            replace=False)

                # ToDo: once everything is sft, use new function:
                subject_streamlines = split_streamlines(sft, self.rng,
                                                        split_ids)

            # Remember the indices of the Y sub-batch
            subbatcht_start = len(batch_streamlines)
            subbatcht_end = subbatcht_start + len(subject_streamlines)
            batch_subj_to_y_ids_processed[subj] = \
                slice(subbatcht_start, subbatcht_end)

            # Add Y streamlines to batch
            batch_streamlines.extend(subject_streamlines)

        # Flip half of streamline batch
        # You could want to flip ALL your data and then use both the initial
        # data and flipped data. But this would take twice the memory. Here, for
        # each epoch, you have 50% chance to be flipped. If you train for enough
        # epochs, high chance that you will have used both directions of your
        # streamline at least once. A way to absolutely ensure using both
        # directions the same number of time, we could use a flag and at each
        # epoch, flip those with unflipped flag. But that adds a bool for each
        # streamline in your dataset and probably not so useful.
        ids = np.arange(len(batch_streamlines))
        self.rng.shuffle(ids)
        flip_ids = ids[:len(ids) // 2]
        batch_streamlines = [flip_streamlines(s) if i in flip_ids else s
                             for i, s in enumerate(batch_streamlines)]
        # ToDo
        #  To change. See new function: sft = reverse_streamlines(sft, flip_ids)

        return batch_streamlines, batch_subj_to_y_ids_processed

    @staticmethod
    def _get_batch_target(batch_streamlines: List[np.ndarray],
                          device: torch.device = None):  # toDo pourquoi device est utile? Pourquoi pas utiliser self.device?
        """
        Get the direction between two adjacent points for each streamlines.
        """
        target = [torch.as_tensor(s[1:] - s[:-1],
                                  dtype=torch.float32,
                                  device=device)
                  for s in batch_streamlines]
        return target

    def _get_batch_x_interp(self, batch_streamlines: List[np.ndarray],
                            batch_subj_to_y_ids_processed: Dict[int, slice],
                            batch_directions,  # ?type?
                            device: torch.device = None):
        """
        Get the DWI (depending on volume: as raw, SH, fODF, etc.) volume for
        each point in each streamline (+ depending on options: neighborhood,
        and preceding diretion)
        """
        batch_x_data = []
        for subj, y_ids in batch_subj_to_y_ids_processed.items():
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
>>>>>>> 4eb53f0... Batch_sampler now in model. Separated model-dependant functions from multisubject class and into batch_sampler. Arranged multisubject to include groups from config file.
