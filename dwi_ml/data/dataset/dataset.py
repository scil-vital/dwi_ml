# -*- coding: utf-8 -*-
"""
See also lazy version in lazy_multi_subject_dataset file.
Separated because this file was becoming HUGE!
"""

import logging
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Union

import h5py
from nibabel.streamlines import ArraySequence
import numpy as np
from scilpy.tracking.tools import resample_streamlines_step_size
import torch
import torch.multiprocessing
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import Dataset
import tqdm

from dwi_ml.cache.cache_manager import SingleThreadCacheManager
from dwi_ml.data.dataset.data_list import (DataListForTorch,
                                           LazyDataListForTorch)
from dwi_ml.data.dataset.parameter_description import PARAMETER_DESCRIPTION
from dwi_ml.data.dataset.single_subject_containers import (SubjectData,
                                                           LazySubjectData)
from dwi_ml.data.processing.streamlines.data_augmentation import (
    reverse_streamlines,
    add_noise_to_streamlines,
    split_streamlines)


class MultiSubjectDataset(Dataset):
    """Dataset containing multiple SubjectData objects saved in a DataList.
    Based on torch's dataset class. Provides functions for a DataLoader to
    iterate over data and process batches.
    """
    def __init__(self, path: str, rng: np.random.RandomState, name: str = None,
                 use_streamline_noise: bool = False, step_size: float = None,
                 neighborhood_dist_mm: float = None, nb_neighborhood_axes=6,
                 streamlines_cut_ratio: float = None,
                 add_previous_dir: bool = False, do_interpolation: bool = False,
                 device: torch.device = torch.device('cpu'),
                 taskman_managed: bool = False):

        # See parameters description
        self.description = PARAMETER_DESCRIPTION

        # Dataset info
        self.path = path
        self.name = name if name else os.path.basename(self.path)

        # Concerning the choice of streamlines:
        self.use_streamline_noise = use_streamline_noise
        self.step_size = step_size
        self.streamlines_cut_ratio = streamlines_cut_ratio
        self.do_interpolation = do_interpolation
        self.default_noise_mm = 0.1 # Default noise variance for compressed
                                    # streamlines, otherwise 0.1 * step-size

        # Concerning the choice of dMRI data
        self.sh_order = None  # Stored in the dataset file

        # Concerning the use of inputs
        self.add_previous_dir = add_previous_dir
        self.add_neighborhood_mm = neighborhood_dist_mm
        self.neighborhood_vectors = None  # Will be computed later
        self.nb_neighborhood_axes = nb_neighborhood_axes

        # Concerning the memory usage:
        self.rng = rng
        self.device = device
        self.taskman_managed = taskman_managed

        # Preparing the dataset
        self.data_list = None  # type will be DataListForTorch or, in lazy version, LazyDataListForTorch
        self.subjID_to_streamlineID = None
        self.total_streamlines = None
        self.streamline_timesteps = None

    def load(self):
        """Load raw dataset into memory."""
        with h5py.File(self.path, 'r') as hdf_file:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            database_version = hdf_file.attrs["version"]
            logging.info("Loaded hdf file: {}".format(self.path))
            logging.info("Database version: {}".format(database_version))
            self.sh_order = int(hdf_file.attrs["sh_order"])
            keys = sorted(list(hdf_file.keys()))

            # Build manager and initialize values
            self.data_list = self.build_data_list(hdf_file)
            self.subjID_to_streamlineID = OrderedDict()
            self.total_streamlines = 0
            streamline_timesteps_list = []

            # Using tqdm progress bar, load all subjects from hdf_file
            for subject_key in tqdm.tqdm(keys, ncols=100,
                                         disable=self.taskman_managed):
                # Add subject's data to the list
                subj_data = self.create_subject_from_hdf(hdf_file, subject_key)
                subjid = self.data_list.add_subject(subj_data)

                # Gestion of streamline is different for lazy or not so
                # encapsulated.
                streamline_timesteps_list = self.arrange_streamlines(
                    subj_data, subjid, streamline_timesteps_list)

        # Arrange final data properties
        self.streamline_timesteps = np.concatenate(streamline_timesteps_list,
                                                   axis=0)

        # Get neighborhood information.
        if self.add_neighborhood_mm:
            affine_vox2rasmm = \
                self.data_list[0].dmri_data.affine_vox2rasmm
            self.neighborhood_vectors = get_interp_neighborhood_vectors(
                self.add_neighborhood_mm, nb_axes=self.nb_neighborhood_axes,
                convert_mm_to_vox=True, affine=affine_vox2rasmm)

    def arrange_streamlines(self, subj_data, subjid, streamline_timesteps_list,
                            hdf_file=None):
        """hdf_file not used. But added to keep same signature as lazy version. """

        # Assign a unique ID to every streamline
        n_streamlines = len(subj_data.streamlines)
        self.subjID_to_streamlineID[subjid] = (
            self.total_streamlines, self.total_streamlines + n_streamlines)

        # Update total nb of streamlines in the dataset
        self.total_streamlines += n_streamlines

        # Get number of timesteps per streamline
        streamline_timesteps_list.append(
            np.array(subj_data.streamlines.get_lengths(), dtype=np.int16))

        return streamline_timesteps_list

    def get_subject_data(self, subj_id):
        """Contains both dMRI data and streamlines. Different in lazy version"""
        return self.data_list[subj_id]

    def get_subject_dmri_data(self, subj_id):
        """Different in lazy version"""
        return self.get_subject_data(subj_id).dmri_data.as_tensor

    def get_subject_streamlines_subset(self, subj_id, ids):
        """Same in lazy version"""
        return self.get_subject_data(subj_id).streamlines[ids]

    @staticmethod
    def build_data_list(hdf_file):
        # hdf_file not used. But this is used by load, and the lazy version uses
        # this parameter. Keeping the same signature.
        return DataListForTorch()

    @staticmethod
    def create_subject_from_hdf(hdf_file, subject_id):
        """Different in lazy version"""
        return SubjectData.create_from_hdf(hdf_file[subject_id])

    def __len__(self):
        return self.total_streamlines

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        """
        - If idx: int, it is a unique item database streamline identifier.
            Then, returns a (streamline_id, subject_id) associated with it.
        - If idx: tuple, it is already a (streamline_id, subj_id). Then this
            function does nothing but necessary. Will be used by torch.
        """
        if type(idx) is tuple:
            return idx
        else:
            for subjid, (start, end) in self.subjID_to_streamlineID.items():
                if start <= idx < end:
                    streamlineid = idx - start
                    return streamlineid, subjid
            raise ValueError("Could not find (streamlineid, subjid) "
                             "from idx: {}".format(idx))

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
                          device: torch.device = None):                                                         # toDo pourquoi device est utile? Pourquoi pas utiliser self.device?
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
            data_volume = self.get_subject_dmri_data(subj)
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


class LazyMultiSubjectDataset(MultiSubjectDataset):
    """Lazy dataset containing multiple TractographyData objects."""

    def __init__(self, *args, cache_size: int = 0, **kwargs):
        # Most parameters are the same as non-lazy
        super().__init__(*args, **kwargs)

        # In case `None` was passed explicitly, change cache_size:
        self.cache_size = cache_size or 0

        # This is important. HDF5 file opening should be done in the
        # __get_item__ method or similar, AFTER the worker sub-processes
        # have started. This way, all sub-processes will have their own
        # HDF handle. Otherwise, there may be data corruption because
        # h5py is not thread-safe.
        self.hdf_handle = None
        if self.cache_size > 0:
            self.volume_cache_manager = None

    def arrange_streamlines(self, subj_data, subjid, streamline_timesteps_list,
                            hdf_file=None):
        # Find his streamlines information
        tmp_subj_data_loaded = self.data_list[(subjid, hdf_file)]

        # Assign a unique ID to every streamline
        n_streamlines = len(tmp_subj_data_loaded.streamlines)
        self.subjID_to_streamlineID[subjid] = (
            self.total_streamlines, self.total_streamlines + n_streamlines)

        # Get number of timesteps per streamlines
        streamline_timesteps_list.append(
            np.array(tmp_subj_data_loaded.streamlines.get_lengths(), dtype=np.int16))

        return streamline_timesteps_list

    def get_subject_data(self, item):
        """Contains both dMRI data and streamlines."""
        if self.hdf_handle is None:
            self.hdf_handle = h5py.File(self.path, 'r')
        return self.data_list[(item, self.hdf_handle)]

    def get_subject_dmri_data(self, subj_id):
        if self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                volume_data = self.volume_cache_manager[subj_id]
            except KeyError:
                # volume_data isn't cached; fetch and cache it
                volume_data = self.get_subject_data(subj_id).dmri_data.as_tensor

                # Send volume_data on device and keep it there while it's cached
                volume_data = volume_data.to(self.device)

                self.volume_cache_manager[subj_id] = volume_data
            return volume_data
        else:
            # No cache is used
            return self.get_subject_data(subj_id).dmri_data.as_tensor

    @staticmethod
    def build_data_list(hdf_file):
        assert hdf_file is not None
        return LazyDataListForTorch(hdf_file)

    @staticmethod
    def create_subject_from_hdf(hdf_file, subject_id):
        return LazySubjectData(subject_id=subject_id)

    def __del__(self):
        if self.hdf_handle is not None:
            self.hdf_handle.close()

    # Functions usign super's:
    # get_subject_streamlines_subset
    # __len__
    # __getitem__
    # get_batch_x_y_and_target (=COLLATE_FN)


