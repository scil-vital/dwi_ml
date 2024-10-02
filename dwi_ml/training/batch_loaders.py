# -*- coding: utf-8 -*-
"""
                                Batch sampler

These classes define how to sample the streamlines available in the
MultiSubjectData.

AbstractBatchLoader:

- Defines the load_batch method:
    - Loads the streamlines associated to sampled ids. Can resample them.

    - Performs data augmentation (on-the-fly to avoid having to multiply data
     on disk) (ex: splitting, reversing, adding noise).

    NOTE: Actual loaded batch size might be different from `batch_size`
    depending on chosen data augmentation. This sampler takes streamline
    cutting and resampling into consideration, and as such, will return more
    (or less) points than the provided `batch_size`.

----------
                        Implemented child classes

BatchLoaderOneInput:

- Defines the load_batch_inputs method:
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
                                collate_fn=batch_loader.load_batch)
"""

from collections import defaultdict
import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.data.processing.streamlines.data_augmentation import (
    reverse_streamlines, split_streamlines, resample_or_compress)
from dwi_ml.data.processing.utils import add_noise_to_tensor
from dwi_ml.models.main_models import MainModelOneInput, \
    ModelWithNeighborhood, MainModelAbstract

logger = logging.getLogger('batch_loader_logger')


class DWIMLStreamlinesBatchLoader:
    def __init__(self, dataset: MultiSubjectDataset, model: MainModelAbstract,
                 streamline_group_name: str, rng: int,
                 split_ratio: float = 0.,
                 noise_gaussian_size_forward: float = 0.,
                 noise_gaussian_size_loss: float = 0.,
                 reverse_ratio: float = 0., log_level=logging.root.level):
        """
        Parameters
        ----------
        dataset : MultisubjectSubset
            Dataset to sample from.
        streamline_group_name: str
            The name of the group to use to load the sequences among all
            streamline_groups in the data_source.
        rng : int
            Seed for the random number generator.
        split_ratio : float
            DATA AUGMENTATION: Percentage of streamlines to randomly split
            into 2, in each batch (keeping both segments as two independent
            streamlines). Default = 0.
            The reason for cutting is to help the ML algorithm to track from
            the middle of WM by having already seen half-streamlines. If you
            are using interface seeding, this is not necessary.
        noise_gaussian_size_forward : float
            DATA AUGMENTATION: Add random Gaussian noise to streamline
            coordinates with given variance. Noise is added AFTER
            interpolation of underlying data.
            This corresponds to the std of the Gaussian.
            Value is given in voxel world. Noise is truncated to
            +/- 2*noise_gaussian_size.
            ** Suggestion. Make sure that
                     2*(noise_gaussian_size) < step_size/2 (in vox)
            to avoid flipping direction. In the worst case, the starting point
            of a segment may advance of step_size/2 while the ending point
            rewinds of step_size/2, but not further, so the direction of the
            segment won't flip. Suggestion, you could choose ~0.1 * step-size.
            Default = 0.
        noise_gaussian_size_forward : float
            Idem, for streamlines used as target (during training only).
        reverse_ratio: float
            DATA AUGMENTATION: If set, reversed a part of the streamlines in
            the batch. You could want to reverse ALL your data and then use
            both the initial data and reversed data. But this would take twice
            the memory. If your ratio is, say, 0.5, streamlines have a 50%
            chance to be reversed. If you train for enough epochs, high chance
            that you will have used both directions of your streamline at least
            once. Default: 0.
            A way to absolutely ensure using both directions the same number of
            time, we could use a flag and at each epoch, reverse those with
            unreversed flag. But that adds a bool for each streamline in your
            dataset and probably not so useful.
        """
        # Batch sampler variables
        self.dataset = dataset
        self.model = model
        self.streamline_group_name = streamline_group_name

        # Find idx of streamline group
        self.streamline_group_idx = self.dataset.streamline_groups.index(
            self.streamline_group_name)
        self.data_contains_connectivity = \
            self.dataset.streamlines_contain_connectivity[
                self.streamline_group_idx]

        # Set random numbers
        self.rng = rng
        self.np_rng = np.random.RandomState(self.rng)

        # Data augmentation for streamlines:
        self.noise_gaussian_size_forward = noise_gaussian_size_forward
        self.noise_gaussian_size_loss = noise_gaussian_size_loss
        self.split_ratio = split_ratio
        self.reverse_ratio = reverse_ratio
        if self.split_ratio and not 0 <= self.split_ratio <= 1:
            raise ValueError('Split ratio must be a float between 0 and 1.')
        if self.reverse_ratio and not 0 <= self.reverse_ratio <= 1:
            raise ValueError('Reverse ration must be a float between 0 and 1')

        logger.setLevel(log_level)

        # For later use, context
        self.context = None
        self.context_subset = None
        self.context_noise_size_forward = None
        self.context_noise_size_loss = None

    @property
    def params_for_checkpoint(self):
        """
        All parameters. Contains at least all parameters that would be
        necessary to create this batch sampler again (except the dataset).
        """
        params = {
            'streamline_group_name': self.streamline_group_name,
            'rng': self.rng,
            'noise_gaussian_size_forward': self.noise_gaussian_size_forward,
            'noise_gaussian_size_loss': self.noise_gaussian_size_loss,
            'reverse_ratio': self.reverse_ratio,
            'split_ratio': self.split_ratio
        }
        return params

    @classmethod
    def init_from_checkpoint(cls, dataset, model, checkpoint_state,
                             new_log_level=None):
        if new_log_level is not None:
            batch_loader = cls(dataset=dataset, model=model,
                               log_level=new_log_level, **checkpoint_state)
        else:
            batch_loader = cls(dataset=dataset, model=model,
                               **checkpoint_state)
        return batch_loader

    def set_context(self, context: str):
        if self.context != context:
            if context == 'training':
                self.context_subset = self.dataset.training_set
                self.context_noise_size_forward = self.noise_gaussian_size_forward
                self.context_noise_size_loss = self.noise_gaussian_size_loss
            elif context == 'validation':
                self.context_subset = self.dataset.validation_set
                self.context_noise_size_forward = 0.
                self.context_noise_size_loss = 0.
            else:
                raise ValueError("Context should be either 'training' or "
                                 "'validation'.")
            self.context = context
        self.dataset.context = context

    def _data_augmentation_sft(self, sft):
        if self.model.step_size is not None and \
                self.context_subset.step_size == self.model.step_size:
            logger.debug("Step size is the same as when creating "
                         "the hdf5 dataset. Not resampling again.")
        elif self.model.compress_lines is not None and \
                self.context_subset.compress == self.model.compress_lines:
            logger.debug("Compression rate is the same as when creating "
                         "the hdf5 dataset. Not compressing again.")
        elif self.model.nb_points is not None and self.model.nb_points == self.context_subset.nb_points:
            logging.debug("Number of points per streamline is the same"
                          " as when creating the hdf5. Not resampling again.")
        else:
            logger.debug("Resample streamlines using: \n" +
                         "- step_size: {}\n".format(self.model.step_size) +
                         "- compress_lines: {}".format(self.model.compress_lines) +
                         "- nb_points: {}".format(self.model.nb_points))
            sft = resample_or_compress(sft, self.model.step_size,
                                       self.model.nb_points,
                                       self.model.compress_lines)

        # Splitting streamlines
        # This increases the batch size, but does not change the total
        # length
        if self.split_ratio and self.split_ratio > 0:
            logger.debug("            Splitting: {}".format(self.split_ratio))
            all_ids = np.arange(len(sft))
            n_to_split = int(np.floor(len(sft) * self.split_ratio))
            split_ids = self.np_rng.choice(all_ids, size=n_to_split,
                                           replace=False)
            sft = split_streamlines(sft, self.np_rng, split_ids)

        # Reversing streamlines
        if self.reverse_ratio and self.reverse_ratio > 0:
            logger.debug("            Reversing: {}"
                         .format(self.reverse_ratio))
            ids = np.arange(len(sft))
            self.np_rng.shuffle(ids)
            reverse_ids = ids[:int(len(ids) * self.reverse_ratio)]
            sft = reverse_streamlines(sft, reverse_ids)

        return sft

    def add_noise_streamlines_forward(self, batch_streamlines, device):
        # This method is called by the trainer only before the forward method.
        # Targets are not modified for the loss computation.
        # Adding noise to coordinates. Streamlines are in voxel space by now.
        # Noise is considered in voxel space.
        if (self.context_noise_size_forward is not None and
                self.context_noise_size_forward > 0):
            logger.debug("            Adding noise {}"
                         .format(self.context_noise_size_forward))
            batch_streamlines = add_noise_to_tensor(
                batch_streamlines, self.context_noise_size_forward, device)
        return batch_streamlines

    def add_noise_streamlines_loss(self, batch_streamlines, device):
        # This method is called by the trainer only before the forward method.
        # Targets are not modified for the loss computation.
        # Adding noise to coordinates. Streamlines are in voxel space by now.
        # Noise is considered in voxel space.
        if (self.context_noise_size_loss is not None and
                self.context_noise_size_loss > 0):
            logger.debug("            Adding noise {}"
                         .format(self.context_noise_size_loss))
            batch_streamlines = add_noise_to_tensor(
                batch_streamlines, self.context_noise_size_loss, device)
        return batch_streamlines

    def load_batch_streamlines(
            self, streamline_ids_per_subj: List[Tuple[int, list]]):
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
            - batch_streamlines: list[torch.tensor]
                The new streamlines after data augmentation, IN VOXEL SPACE,
                CORNER.
            - final_s_ids_per_subj: Dict[int, slice]
                The new streamline ids per subj in this augmented batch.
        """
        if self.context is None:
            raise ValueError("Context must be set prior to using the batch "
                             "loader.")

        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        final_s_ids_per_subj = defaultdict(slice)
        batch_streamlines = []
        for subj, s_ids in streamline_ids_per_subj:
            logger.debug(
                "            Data loader: Processing data preparation for "
                "subj {} (preparing {} streamlines)".format(subj, len(s_ids)))

            # No cache for the sft data. Accessing it directly.
            # Note: If this is used through the dataloader, multiprocessing
            # is used. Each process will open a handle.
            subj_data = \
                self.context_subset.subjs_data_list.get_subj_with_handle(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # Get streamlines as sft
            logger.debug("            Loading sampled streamlines...")
            sft = subj_sft_data.as_sft(s_ids)
            sft = self._data_augmentation_sft(sft)

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

        batch_streamlines = [torch.as_tensor(s) for s in batch_streamlines]

        return batch_streamlines, final_s_ids_per_subj

    def load_batch_streamlines_and_related(
            self, streamline_ids_per_subj: List[Tuple[int, list]]):
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
            - batch_streamlines: list[torch.tensor]
                The new streamlines after data augmentation, IN VOXEL SPACE,
                CORNER.
            - final_s_ids_per_subj: Dict[int, slice]
                The new streamline ids per subj in this augmented batch.
        """
        if self.context is None:
            raise ValueError("Context must be set prior to using the batch "
                             "loader.")

        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        final_s_ids_per_subj = defaultdict(slice)
        batch_streamlines = []
        streamlines_related_data = []
        for subj, s_ids in streamline_ids_per_subj:
            logger.debug(
                "            Data loader: Processing data preparation for "
                "subj {} (preparing {} streamlines)".format(subj, len(s_ids)))

            # No cache for the sft data. Accessing it directly.
            # Note: If this is used through the dataloader, multiprocessing
            # is used. Each process will open a handle.
            subj_data = \
                self.context_subset.subjs_data_list.get_subj_with_handle(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # Get streamlines as sft
            logger.debug("            Loading sampled streamlines...")
            sft = subj_sft_data.as_sft(s_ids)

            # TODO: modify this list consequently to the data augmentations.
            # Currently, if the data augmentation adds/removes streamlines,
            # the related data won't match the streamlines list anymore.
            related_data = subj_sft_data.get_related_data(
                s_ids)  # Can return None
            sft = self._data_augmentation_sft(sft)

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

            if related_data is not None:
                streamlines_related_data.extend(related_data)

        batch_streamlines = [torch.as_tensor(s) for s in batch_streamlines]

        if len(streamlines_related_data) > 0:
            assert len(streamlines_related_data) == len(batch_streamlines), \
                "Related data should have the same length as the streamlines."

        return batch_streamlines, final_s_ids_per_subj, streamlines_related_data

    def load_batch_connectivity_matrices(
            self, streamline_ids_per_subj: Dict[int, slice]):
        if not self.data_contains_connectivity:
            raise ValueError("No connectivity matrix in this dataset.")

        # The batch's streamline ids will change throughout processing because
        # of data augmentation, so we need to do it subject by subject to
        # keep track of the streamline ids. These final ids will correspond to
        # the loaded, processed streamlines, not to the ids in the hdf5 file.
        subjs = list(streamline_ids_per_subj.keys())
        nb_subjs = len(subjs)
        matrices = [None] * nb_subjs
        volume_sizes = [None] * nb_subjs
        connectivity_nb_blocs = [None] * nb_subjs
        connectivity_labels = [None] * nb_subjs
        for i, subj in enumerate(subjs):
            # No cache for the sft data. Accessing it directly.
            # Note: If this is used through the dataloader, multiprocessing
            # is used. Each process will open a handle.
            subj_data = \
                self.context_subset.subjs_data_list.get_subj_with_handle(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # We could access it only at required index, maybe. Loading the
            # whole matrix here.
            (matrices[i], volume_sizes[i],
             connectivity_nb_blocs[i], connectivity_labels[i]) = \
                subj_sft_data.get_connectivity_matrix_and_info()

        return (matrices, volume_sizes,
                connectivity_nb_blocs, connectivity_labels)


class DWIMLBatchLoaderOneInput(DWIMLStreamlinesBatchLoader):
    """
    Loads:
        input = one volume group
                (data underlying each point of the streamline)
                (possibly with its neighborhood)
        target = the whole streamlines as sequences.
    """
    model: MainModelOneInput

    def __init__(self, input_group_name, **kw):
        """
        Params
        ------
        input_group_name: str
            Name of the input group in the hdf5 dataset.
        """
        super().__init__(**kw)

        self.input_group_name = input_group_name
        self.use_neighborhood = isinstance(self.model, ModelWithNeighborhood)

        # Find group index in the data_source
        try:
            idx = self.dataset.volume_groups.index(input_group_name)
        except ValueError:
            raise ValueError("Required input group {} is not in list of "
                             "volume groups: {}"
                             .format(input_group_name,
                                     self.dataset.volume_groups))
        self.input_group_idx = idx

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'input_group_name': self.input_group_name,
        })
        return p

    @property
    def states(self):
        states = {
            'rng_state': self.np_rng.get_state(),
        }
        return states

    def load_batch_inputs(self, batch_streamlines: List[torch.tensor],
                          streamline_ids_per_subj: Dict[int, slice]):
        """
        Get the DWI (depending on volume: as raw, SH, fODF, etc.) volume for
        each point in each streamline (+ depending on options: neighborhood,
        and preceding diretion).

        Current context must be set.

        Params
        ------
        batch_streamlines: list[np.array]
            The streamlines (after data augmentation) in voxel space, with
            corner origin.
        streamline_ids_per_subj: Dict[int, slice]
            The ids corresponding to each subject (so we can load the
            associated subject's input volume).
        save_batch_input_mask: bool
            Debugging purposes. Saves the input coordinates as a mask.
            The inputs will be modified to a tuple containing the
            batch_streamlines, to compare the streamlines with masks.
        device: torch device
            Torch device.

        Returns
        -------
        batch_x_data : List[tensor]
            The list of tensors inputs for each streamline. Each tensor is of
            shape [nb points, nb_features].
        """
        batch_x_data = []

        for subj, y_ids in streamline_ids_per_subj.items():
            logger.debug("            Data loader: loading input volume.")

            streamlines = batch_streamlines[y_ids]

            # Trilinear interpolation uses origin=corner, vox space, but ok
            # because in load_batch, we use sft.to_vox and sft.to_corner
            # before adding streamline to batch.
            subbatch_x_data = self.model.prepare_batch_one_input(
                streamlines, self.context_subset, subj,
                self.input_group_idx)

            batch_x_data.extend(subbatch_x_data)

        return batch_x_data
