# -*- coding: utf-8 -*-
"""
                                Batch sampler

These classes defines how to sample the streamlines available in the
MultiSubjectData.

AbstractBatchSampler:

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
                                collate_fn=batch_loader.load_batch)
"""

from collections import defaultdict
import logging
from typing import Dict, List, Tuple

from dipy.io.stateful_tractogram import StatefulTractogram
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
from scilpy.utils.streamlines import compress_sft
import torch
import torch.multiprocessing

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.data.processing.streamlines.data_augmentation import (
    add_noise_to_streamlines, reverse_streamlines, split_streamlines)
from dwi_ml.models.main_models import MainModelOneInput

logger = logging.getLogger('batch_loader_logger')


class DWIMLAbstractBatchLoader:
    def __init__(self, dataset: MultiSubjectDataset,
                 streamline_group_name: str, rng: int,
                 step_size: float = None, compress: bool = False,
                 split_ratio: float = 0.,
                 noise_gaussian_size_training: float = 0.,
                 noise_gaussian_var_training: float = 0.,
                 noise_gaussian_size_validation: float = None,
                 noise_gaussian_var_validation: float = None,
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
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). Default: None (train on streamlines as they are).
            Note that you probably already fixed a step size when
            creating your dataset, but you could use a different one here if
            you wish.
        compress: bool
            If true, compress streamlines. Cannot be used together with
            step_size. Once again, the choice can be different in the batch
            sampler than chosen when creating the hdf5. Default: False.
        split_ratio : float
            DATA AUGMENTATION: Percentage of streamlines to randomly split
            into 2, in each batch (keeping both segments as two independent
            streamlines). Default = 0.
            The reason for cutting is to help the ML algorithm to track from
            the middle of WM by having already seen half-streamlines. If you
            are using interface seeding, this is not necessary.
        noise_gaussian_size_training : float
            DATA AUGMENTATION: Add random Gaussian noise to streamline
            coordinates with given variance. This corresponds to the std of the
            Gaussian. If step_size is not given, make sure it is smaller than
            your step size to avoid flipping direction. Ex, you could choose
            0.1 * step-size. Noise is truncated to +/- 2*noise_sigma and to
            +/- 0.5 * step-size (if given). Default = 0.
        noise_gaussian_var_training: float
            DATA AUGMENTATION: If this is given, a variation is applied to the
            streamline_noise_gaussian_size to have more noisy streamlines and
            less noisy streamlines. This means that the real gaussian_size will
            be a random number between [size - var, size + var].
            Default = 0.
        noise_gaussian_size_validation : float or None
            Same as training
        noise_gaussian_var_validation: float or None
            Same as training
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
        self.streamline_group_name = streamline_group_name
        if step_size and compress:
            raise ValueError("You may choose either resampling or compressing,"
                             "but not both.")
        elif step_size and step_size <= 0:
            raise ValueError("Step size can't be 0 or less!")
            # Note. When using
            # scilpy.tracking.tools.resample_streamlines_step_size, a warning
            # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
            # that the value is suspicious. Not raising the same warnings here
            # as you may be wanting to test weird things to understand better
            # your model.
        self.step_size = step_size
        self.compress = compress

        # Find idx of streamline group
        self.streamline_group_idx = self.dataset.streamline_groups.index(
            self.streamline_group_name)

        # Set random numbers
        self.rng = rng
        self.np_rng = np.random.RandomState(self.rng)
        torch.manual_seed(self.rng)  # Set torch seed

        # Data augmentation for streamlines:
        self.noise_gaussian_size_train = noise_gaussian_size_training
        self.noise_gaussian_var_train = noise_gaussian_var_training
        self.noise_gaussian_size_valid = noise_gaussian_size_validation
        self.noise_gaussian_var_valid = noise_gaussian_var_validation
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
        self.context_noise_size = None
        self.context_noise_var = None

    @property
    def params_for_checkpoint(self):
        """
        All parameters. Contains at least all parameters that would be
        necessary to create this batch sampler again (except the dataset).
        """
        params = {
            'streamline_group_name': self.streamline_group_name,
            'rng': self.rng,
            'noise_gaussian_size_training': self.noise_gaussian_size_train,
            'noise_gaussian_var_training': self.noise_gaussian_var_train,
            'noise_gaussian_size_validation': self.noise_gaussian_size_valid,
            'noise_gaussian_var_validation': self.noise_gaussian_var_valid,
            'reverse_ratio': self.reverse_ratio,
            'split_ratio': self.split_ratio,
            'step_size': self.step_size,
            'compress': self.compress,
        }
        return params

    @property
    def params_for_json_prints(self):
        # All params are all right.
        p = self.params_for_checkpoint
        p.update({'type': str(type(self))})
        return p

    def set_context(self, context: str):
        if self.context != context:
            if context == 'training':
                self.context_subset = self.dataset.training_set
                self.context_noise_size = self.noise_gaussian_size_train
                self.context_noise_var = self.noise_gaussian_var_train
            elif context == 'validation':
                self.context_subset = self.dataset.validation_set
                self.context_noise_size = self.noise_gaussian_size_valid
                self.context_noise_var = self.noise_gaussian_var_valid
            else:
                raise ValueError("Context should be either 'training' or "
                                 "'validation'.")
            self.context = context
        self.dataset.context = context

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]]):
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
            # is used. Each process will open an handle.
            subj_data = \
                self.context_subset.subjs_data_list.get_subj_with_handle(subj)
            subj_sft_data = subj_data.sft_data_list[self.streamline_group_idx]

            # Get streamlines as sft
            logger.debug("            Loading sampled streamlines...")
            sft = subj_sft_data.as_sft(s_ids)

            # Resampling streamlines to a fixed step size, if any
            if self.step_size:
                logger.debug("            Resampling: {}"
                             .format(self.step_size))
                if self.context_subset.step_size == self.step_size:
                    logger.debug("Step size is the same as when creating "
                                 "the hdf5 dataset. Not resampling again.")
                else:
                    sft = resample_streamlines_step_size(
                        sft, step_size=self.step_size)

            # Compressing, if wanted.
            if self.compress:
                logger.debug("            Compressing: {}"
                             .format(self.compress))
                sft = compress_sft(sft)

            # Adding noise to coordinates
            # Noise is considered in mm so we need to make sure the sft is in
            # rasmm space
            if self.context_noise_size and self.context_noise_size > 0:
                logger.debug("            Adding noise {} +- {}"
                             .format(self.context_noise_size,
                                     self.context_noise_var))
                sft.to_rasmm()
                sft = add_noise_to_streamlines(sft, self.context_noise_size,
                                               self.context_noise_var,
                                               self.np_rng, self.step_size)

            # Splitting streamlines
            # This increases the batch size, but does not change the total
            # length
            if self.split_ratio and self.split_ratio > 0:
                logger.debug("            Splitting: {}"
                             .format(self.split_ratio))
                all_ids = np.arange(len(sft))
                n_to_split = int(np.floor(len(sft) * self.split_ratio))
                split_ids = self.np_rng.choice(all_ids, size=n_to_split,
                                               replace=False)
                sft = split_streamlines(sft, self.np_rng, split_ids)

            # Reversing streamlines
            if self.reverse_ratio and self.reverse_ratio > 0:
                logger.debug("            Reversing")
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
        logger.debug("            Project-specific data augmentation, if "
                     "any.")

        return sft


class DWIMLBatchLoaderOneInput(DWIMLAbstractBatchLoader):
    """
    Loads:
        input = one volume group
                (data underlying each point of the streamline)
                (possibly with its neighborhood)
        target = the whole streamlines as sequences.
    """
    def __init__(self, input_group_name, model: MainModelOneInput,
                 wait_for_gpu: bool = False, **kw):
        """
        Params
        ------
        input_group_name: str
            Name of the input group in the hdf5 dataset.
        model: ModelOneInput
            The model.
        wait_for_gpu: bool
            If true, will not compute the inputs directly when using
            load_batch. User can call the compute_inputs method himself later
            on. Typically, Dataloader (who call load_batch) uses CPU.
            Default: False
        """
        super().__init__(**kw)

        # toDo. GPU: Would be more logical to send this as params when using
        #  load_batch as collate_fn in the Dataloader during training.
        #  Possible?
        self.wait_for_gpu = wait_for_gpu
        self.input_group_name = input_group_name
        self.model = model

        # Find group index in the data_source
        idx = self.dataset.volume_groups.index(input_group_name)
        self.input_group_idx = idx

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'input_group_name': self.input_group_name,
            # Sending to list to allow json dump
            'wait_for_gpu': self.wait_for_gpu
        })
        return p

    @property
    def states(self):
        states = {
            'rng_state': self.np_rng.get_state(),
        }
        return states

    def load_batch(self, streamline_ids_per_subj: List[Tuple[int, list]],
                   save_batch_input_mask: bool = False):
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
            logger.debug("            Not loading the input data because "
                         "user prefers to do it later on GPU.")

            return batch
        else:
            # At this point batch_streamlines are in voxel space with
            # corner origin.
            batch_streamlines, final_s_ids_per_subj = batch

            # Get the inputs
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
            Debugging purposes. Saves the input coordinates as a mask. Must be
            used together with wait_for_gpu=False. The inputs will be modified
            to a tuple containing the batch_streamlines (with wait_for_gpu set
            to False, the directions only would be returned), to compare the
            streamlines with masks.
        device: torch device
            Torch device.

        Returns
        -------
        batch_x_data : List[tensor]
            The list of tensors inputs for each streamlines. Each tensor is of
            shape [nb points, nb_features].
        """
        batch_x_data = []
        batch_input_masks = []

        for subj, y_ids in streamline_ids_per_subj.items():
            logger.debug("            Data loader: loading input volume.")

            streamlines = batch_streamlines[y_ids]
            # We don't use the last coord because it is used only to
            # compute the last target direction, it's not really an input
            streamlines = [s[:-1] for s in streamlines]

            # Trilinear interpolation uses origin=corner, vox space, but ok
            # because in load_batch, we use sft.to_vox and sft.to_corner
            # before adding streamline to batch.
            subbatch_x_data, input_mask = \
                self.model.prepare_batch_one_input(
                    streamlines, self.context_subset, subj,
                    self.input_group_idx, self.neighborhood_points, device)

            batch_x_data.extend(subbatch_x_data)

            if save_batch_input_mask:
                print("DEBUGGING MODE. Returning batch_streamlines "
                      "and mask together with inputs.")
                batch_input_masks.append(input_mask)

                return batch_input_masks, batch_x_data

        return batch_x_data
