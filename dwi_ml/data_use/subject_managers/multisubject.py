"""
We expect the classes here to be used in a file such as model_and_training.py

Here, we define:
    class _MultiSubjectDataManager:
        Public variables and functions:
            .volume_feature_size
            .add_subject()
    class _LazyMultiSubjectDataManager

    ====
    class MultiSubjectDataset: extends torch's dataset class
        Public variables and functions:
            .load()
            .get_batch_x_and_y()
    class LazyMultiSubjectDataset
    class BatchSampler: extends torch's sampler class
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict, defaultdict
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
import torch.multiprocessing
import tqdm
from nibabel.streamlines import ArraySequence
from torch.nn.utils.rnn import PackedSequence, pack_sequence
from torch.utils.data import Dataset, Sampler

from scil_vital.shared.code.data.singlesubject import (
    LazySubjectData, SubjectData)
from scil_vital.shared.code.io.cache_manager import (
    SingleThreadCacheManager)
from scil_vital.shared.code.signal.interpolation import (
    torch_trilinear_interpolation, get_interp_neighborhood_vectors,
    extend_coords_with_interp_neighborhood_vectors)
from scil_vital.shared.code.transformation.streamline import (
    flip_streamline)
from scil_vital.shared.code.transformation.streamlines import (
    add_noise_to_streamlines, cut_random_streamlines,
    resample_streamlines)


class _MultiSubjectDataManager(object):
    """Manage multiple tracto data objects. Everything is loaded into memory
    until it is needed. Will be used by MultiSubjectDataset, lower."""

    def __init__(self):
        # Feature size should be common to all subjects.
        self._subjects_data_list = []
        self.feature_size = None

    @property
    def volume_feature_size(self):
        """Returns the nb of information per voxel in the input dMRI volume."""
        if self.feature_size is None:
            try:
                self.feature_size = \
                    int(self._subjects_data_list[0].dmri_volume.shape[-1])
            except IndexError:
                # No volume has been registered yet. Do not raise an exception,
                # but return 0 as the feature size.
                self.feature_size = 0
        return self.feature_size

    def add_subject(self, subject_data):
        """Adds subject's data to subjects_data_list.
        Returns idx of where subject is inserted.
        """

        # Add subject
        subject_idx = len(self._subjects_data_list)
        self._subjects_data_list.append(subject_data)

        # Make sure all volumes have the same feature size
        assert self.volume_feature_size == subject_data.dmri_volume.shape[-1], \
            "Tried to add a subject whose dMRI volume's feature size was " \
            "different from previous!"
        return subject_idx

    def __getitem__(self, subject_idx):
        """ Necessary for torch"""
        return self._subjects_data_list[subject_idx]

    def __len__(self):
        return len(self._subjects_data_list)


class _LazyMultiSubjectDataManager(_MultiSubjectDataManager):
    def __init__(self, default_hdf_handle):
        super().__init__()
        self._hdf_handle = default_hdf_handle

    @property
    def volume_feature_size(self):
        """Overriding super's function"""
        if self._feature_size is None:
            try:
                self._feature_size = \
                    int(self.__getitem__((0, self._hdf_handle)
                                         ).dmri_volume.shape[-1])
            except IndexError:
                # No volume has been registered yet. Do not raise an exception,
                # but return 0 as the feature size.
                self._feature_size = 0
        return self._feature_size

    def add_subject(self, subject_data):
        """Overriding super's function"""

        data_idx = len(self._subjects_data_list)
        self._subjects_data_list.append(subject_data)

        # Make sure all volumes have the same feature size
        new_subject = self.__getitem__((-1, self._hdf_handle))
        assert self.volume_feature_size == new_subject.dmri_volume.shape[-1], \
            "Tried to add a tractogram whose dMRI volume's feature size was " \
            "different from previous!"

        return data_idx

    def __getitem__(self, subject_item):
        """Overriding super's function"""
        assert type(subject_item) == tuple, \
            "Trying to get an item, but item should be a tuple."
        subject_idx, subject_hdf_handle = subject_item
        partial_subjectdata = self._subjects_data_list[subject_idx]
        return partial_subjectdata.with_handle(subject_hdf_handle)


class MultiSubjectDataset(Dataset):
    """Dataset containing multiple TractographyData objects. Based on torch's
    dataset class. Provides functions for a DataLoader to iterate over data
    and process batches.
    """

    def __init__(self, path: str, rng: np.random.RandomState, name: str = None,
                 add_streamline_noise: bool = False, step_size: float = None,
                 neighborhood_dist_mm: float = None, nb_neighborhood_axes=6,
                 streamlines_cut_ratio: float = None,
                 add_previous_dir: bool = False, do_interpolation: bool = False,
                 device: torch.device = torch.device('cpu'),
                 taskman_managed: bool = False):
        """
        Parameters
        ----------
        path : str
            Path to the processed .hdf5 file.
        rng : np.random.RandomState
            Random number generator.
        name : str
            Name of the dataset. If none, use the basename of the given .hdf5
            file. [None]
        add_streamline_noise : float
            If set, add random gaussian noise to streamline coordinates
            on-the-fly. Noise variance is 0.1 * step-size, or 0.1mm if no step
            size is used. [False]
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). If None, train on streamlines as they are (ex, compressed).
            Note that you probably already fixed a step size when creating your
            dataset, but you could use a different one here if you wish. [None]
        add_neighborhood : float
            Add neighborhood points at the given distance (in mm) in each
            direction (nb_neighborhood_axes). [None] (None and 0 have the same
            effect).
        nb_neighborhood_axes: int
            Nb of axes (directions) to get the neighborhood voxels. This is only
            used if do_interpolation is True. Currently, 6 is the default and
            only implemented version (left, right, front, behind, up, down).
                                                                                                    #ToDO. Raises notImplemented if not 6.
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
        self._path = path
        self._rng = rng
        self._name = name if name else os.path.basename(self._path)
        self._add_streamline_noise = add_streamline_noise
        self._step_size = step_size
        self._streamlines_cut_ratio = streamlines_cut_ratio
        self._add_previous_dir = add_previous_dir
        self._do_interpolation = do_interpolation
        self._device = device
        self._taskman_managed = taskman_managed

        self._add_neighborhood_mm = neighborhood_dist_mm
        self._neighborhood_vectors = None # Will be computed later
        self._nb_neighborhood_axes = nb_neighborhood_axes
        
        self.sh_order = None            # Stored in the dataset file

        self.multisubject_manager = None  # type: _MultiSubjectDataManager
        # Tractodata id to corresponding unique streamline ids:
        self.subjID_to_streamlineID = None
        self._total_streamlines = None
        self.streamline_timesteps = None

        # Default noise variance for compressed streamlines,
        # otherwise 0.1 * step-size
        self._default_noise_mm = 0.1

    def get_subject_data(self, tractodata_id):
        """Contains both volume and streamlines."""
        return self.multisubject_manager[tractodata_id]

    def get_subject_volume(self, tractodata_id):
        return self.get_subject_data(tractodata_id).dmri_volume.data

    def _get_subject_streamlines_subset(self, tractodata_id, ids):
        return self.get_subject_data(tractodata_id).streamlines[ids]

    @staticmethod
    def _build_multisubject_manager(hdf_file):
        """hdf_file not used here but will be necessary in the lazy version.
        Keeping the same signature."""
        return _MultiSubjectDataManager()

    @staticmethod
    def _build_subject_data(hdf_file, subject_key):
        """ Returns one subject data"""
        return SubjectData.create_from_hdf(hdf_file[subject_key])

    def __len__(self):
        return self._total_streamlines

    def __getitem__(self, idx: Union[int, Tuple[int, int]]):
        """Get a (streamline_id, subjectdata_id).

        - If idx: int, it is a unique item index. Unique database streamline
            identifier. Then, return** Not sure if still used. ToDo.
        - If idx: tuple, it is a (streamline id, tractodata id). Then this
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

    def load(self):
        """Load raw dataset into memory."""

        with h5py.File(self._path, 'r') as hdf_file:
            # Load main attributes from hdf file, but each process calling
            # the collate_fn must open its own hdf_file
            database_version = hdf_file.attrs["version"]
            logging.info("Loaded hdf file: {}".format(self._path))
            logging.info("Database version: {}".format(database_version))
            self.sh_order = int(hdf_file.attrs["sh_order"])
            keys = sorted(list(hdf_file.keys()))

            # Build manager and initialize values
            self.multisubject_manager = \
                self._build_multisubject_manager(hdf_file)
            self.subjID_to_streamlineID = OrderedDict()
            self._total_streamlines = 0
            streamline_timesteps_list = []

            # Using tqdm progress bar, load all subjects from hdf_file
            for subject_key in tqdm.tqdm(
                    keys, ncols=100, disable=self._taskman_managed):

                # Add subject's data to the manager
                subj_data = self._build_subject_data(hdf_file, subject_key)
                subjid = self.multisubject_manager.add_subject(subj_data)

                # Assign a unique ID to every streamline
                n_streamlines = len(subj_data.streamlines)
                self.subjID_to_streamlineID[subjid] = (
                    self._total_streamlines,
                    self._total_streamlines + n_streamlines)

                # Update total nb of streamlines in the dataset
                self._total_streamlines += n_streamlines

                # Get number of timesteps per streamline
                # ACCESS TO PROTECTED _lengths! See if dipy changes it one day
                streamline_timesteps_list.append(
                    np.array(subj_data.streamlines._lengths, dtype=np.int16))

        # Arrange final data properties
        self.streamline_timesteps = np.concatenate(streamline_timesteps_list,
                                                   axis=0)

        # Get neighborhood information.
        if self._add_neighborhood_mm:
            affine_vox2rasmm = \
                self.multisubject_manager[0].dmri_volume.affine_vox2rasmm
            self._neighborhood_vectors = get_interp_neighborhood_vectors(
                self._add_neighborhood_mm, nb_axes=self._nb_neighborhood_axes,
                convert_mm_to_vox=True, affine=affine_vox2rasmm)

    def get_batch_x_y_and_target(self, batch_ids: List[Tuple[int, int]]):
        """ Torch uses this function to process the data with the
        dataloader workers = COLLATE_FN. It means that this part is ran on CPU.
        Particularly interesting for interpolation: we don't need to send the
        DWI data to GPU. Interpolation is done on CPU and interpolated data is
        send to GPU for the model_and_training.

        With interpolation:
            Gets the list of (X, target) examples for the model_and_training/validation
            set, with
            - X = the dwi data under each point of the streamlines
            (- Y = the streamlines coordinates. Not returned)
            - target = the "ground truth" direction between each point Y.
            The definition of "dwi data" depends on the chosen model (ex, raw,
            sh, peak, etc.), with interpolation or not, with neighborhood or
            not, with previous direction or not.
        Without interpolation:
            Only gets the list of Y streamlines coordinates. The rest will be
            done by the model_and_training script directly on GPU.                                                ToDo? Make cleaner code? Voir avec Philippe:
                                                                                                         Il m'a expliqué un truc du genre: pour HCP c'est plus efficace comme ça.
                                                                                                         Pour les petits dataset non alors il a laissé l'option.
                                                                                                         Mais on pourrait faire ça par défaut.
                                                                                                         Mettons que je comprends l'explication pour les X data. Mais pourquoi
                                                                                                         le target est géré différemment selon le case???

        Parameters
        ----------
        batch_ids : List of tuple of (int,int)
            A list of (streamline_id, subject_id).

        Returns
        -------
        If self._do_interpolation is False:
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


        if self._do_interpolation:
            # Get the batch target (ground truth directions)
            batch_directions = self._get_batch_target(batch_streamlines,
                                                      device=self._device)

            # Get the batch X (dWI volume under each point of the streamline)
            # NOTE. If we add previous_direction to input X, we don't normalize
            # it. In the case of compressed streamlines, we hope this will give
            # back to the algorithm a sense of distance between point.
            batch_x = self._get_batch_x_interp(batch_streamlines,
                                               batch_subj_to_y_ids_processed,
                                               batch_directions,
                                               device=self._device)

            # Now, normalize targets. If the step size is always the same,
            # shouldn't make any difference. If compressed.... discutable
            # choice.                                                                                               # toDo à discuter en groupe
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
            subject_streamlines = self._get_subject_streamlines_subset(subj,
                                                                       y_ids)

            # Make sure streamlines are stored in a list
            if isinstance(subject_streamlines, ArraySequence):
                subject_streamlines = [s for s in subject_streamlines]

            # Get affine. Used to preprocess streamlines
            affine_vox2rasmm = \
                self.get_subject_data(subj).dmri_volume.affine_vox2rasmm

            # Resample streamlines to a fixed step size
            if self._step_size:
                subject_streamlines = resample_streamlines(
                    subject_streamlines, self._step_size,
                    convert_mm_to_vox=True,
                    affine=affine_vox2rasmm)

            # Add noise to coordinates
                                                                                                # ToDo: add a variance in the distribution of noise between epoques.
                                                                                                #  Comme ça, la même streamline pourra être vue plusieurs fois
                                                                                                #  (dans plsr époques) mais plus ou moins bruitée d'une fois à
                                                                                                #  l'autre.
            if self._add_streamline_noise:
                noise_mm = self._default_noise_mm * (self._step_size or 1.)
                subject_streamlines = add_noise_to_streamlines(
                    subject_streamlines, noise_mm, self._rng,
                    convert_mm_to_vox=True, affine=affine_vox2rasmm)

            # Cut streamlines in random positions, using the same ratio for each
            # Cutting keeps both segments as two independent streamlines, and
            # increases the batch size (but does not change the number of
            # timesteps). Need to do it subject per subject to keep track of the
            # streamline ids.
            if self._streamlines_cut_ratio:
                subject_streamlines = cut_random_streamlines(
                    subject_streamlines, self._streamlines_cut_ratio,
                    rng=self._rng)

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
        self._rng.shuffle(ids)
        flip_ids = ids[:len(ids) // 2]
        batch_streamlines = [flip_streamline(s) if i in flip_ids else s
                             for i, s in enumerate(batch_streamlines)]

        return batch_streamlines, batch_subj_to_y_ids_processed

    @staticmethod
    def _get_batch_target(batch_streamlines: List[np.ndarray],
                          device: torch.device = None):                                             # toDo pourquoi device est utile? Pourquoi pas utiliser self.device?
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
            data_volume = self.get_subject_volume(subj)
            data_volume = data_volume.to(device=device, non_blocking=True)

            # If user chose to add neighborhood:
            if self._add_neighborhood_mm:
                n_input_points = flat_subj_x_coords.shape[0]

                # Extend the coords array with the neighborhood coordinates
                flat_subj_x_coords = \
                    extend_coords_with_interp_neighborhood_vectors(
                        flat_subj_x_coords, self._neighborhood_vectors)

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
                              self._neighborhood_vectors.shape[0])
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
        if self._add_previous_dir:
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

    def load(self):
        """Load raw dataset into memory. Overwriting super's method."""

        with h5py.File(self._path, 'r') as hdf_file:
            # Load main attributes from hdf file, but each process calling
            # the get_batch_x_and_y (collate_fn) must open its own hdf_file
            database_version = hdf_file.attrs["version"]
            logging.info("Loaded hdf file: {}".format(self._path))
            logging.info("Database version: {}".format(database_version))

            self._sh_order = int(hdf_file.attrs["sh_order"])
            keys = sorted(list(hdf_file.keys()))

            self._multisubject_manager = \
                self._build_multisubject_manager(hdf_file)

            self._subjID_to_streamlineID = OrderedDict()
            self._total_streamlines = 0
            streamline_timesteps_list = []

            for subject_key in tqdm.tqdm(keys, ncols=100,
                                         disable=self._taskman_managed):
                # Add subject
                tracto_data = self._build_subject_data(hdf_file, subject_key)
                tractodata_id = \
                    self._multisubject_manager.add_subject(tracto_data)

                # Find his streamlines information
                tractodata_tmp_loaded = \
                    self._multisubject_manager[(tractodata_id, hdf_file)]
                n_streamlines = len(tractodata_tmp_loaded.streamlines)
                self._subjID_to_streamlineID[tractodata_id] = (
                    self._total_streamlines,
                    self._total_streamlines + n_streamlines)
                self._total_streamlines += n_streamlines

                # Get number of timesteps per streamlines
                streamline_timesteps_list.append(
                    np.array(tractodata_tmp_loaded.streamlines.get_lengths(),
                             dtype=np.int16))

            # Concatenate subjects's streamlines
            self._streamline_timesteps = np.concatenate(
                streamline_timesteps_list, axis=0)

            # Get neighborhood
            if self._add_neighborhood_mm:
                s=(0, hdf_file)
                affine_vox2rasmm = \
                    self._multisubject_manager[s].dmri_volume.affine_vox2rasmm
                self._neighborhood_vectors = get_interp_neighborhood_vectors(
                    self._add_neighborhood_mm, convert_mm_to_vox=True,
                    affine=affine_vox2rasmm)

    def get_subject_data(self, item):
        if self.hdf_handle is None:
            self.hdf_handle = h5py.File(self._path, 'r')
        return self._multisubject_manager[(item, self.hdf_handle)]

    def get_subject_volume(self, tractodata_id):
        if self.cache_size > 0:
            # Parallel workers each build a local cache
            # (data is duplicated across workers, but there is no need to
            # serialize/deserialize everything)
            if self.volume_cache_manager is None:
                self.volume_cache_manager = \
                    SingleThreadCacheManager(self.cache_size)

            try:
                # General case: Data is already cached
                volume_data = self.volume_cache_manager[tractodata_id]
            except KeyError:
                # volume_data isn't cached; fetch and cache it
                volume_data = \
                    self.get_subject_data(tractodata_id).dmri_volume.data

                # Send volume_data on device and keep it there while it's cached
                volume_data = volume_data.to(self._device)

                self.volume_cache_manager[tractodata_id] = volume_data
            return volume_data
        else:
            # No cache is used
            return self.get_subject_data(tractodata_id).dmri_volume.data

    @staticmethod
    def _build_multisubject_manager(hdf_file):
        assert hdf_file is not None
        return _LazyMultiSubjectDataManager(hdf_file)

    @staticmethod
    def _build_subject_data(hdf_file, subject_key):
        return LazySubjectData(subject_id=subject_key)

    def __del__(self):
        if self.hdf_handle is not None:
            self.hdf_handle.close()


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

    def __init__(self, data_source: MultiSubjectDataset,                                                    # Says error because it doesn't use super().__init__
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
                    np.arange(len(self.data_source.multisubject_manager)),
                    size=n_volumes, replace=False, p=weights)
            else:
                sampled_tids = self.data_source.subjID_to_streamlineID.keys()
                n_volumes = len(sampled_tids)

            # Compute the number of *original* timesteps required per volume
            # (before resampling)
            n_timesteps_per_volume = self.batch_size / n_volumes

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

                    total_volume_timesteps = 0
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
                        sample_timesteps = \
                            self.data_source.streamline_timesteps[sample_global_ids]

                        volume_batch_fulfilled = False
                        # Keep total volume length under the maximum
                        if (total_volume_timesteps + np.sum(sample_timesteps) >
                                n_timesteps_per_volume):
                            # Select only enough streamlines to fill the
                            # required length
                            cumulative_sum = np.cumsum(sample_timesteps)
                            selected_mask = \
                                cumulative_sum < (n_timesteps_per_volume -
                                                  total_volume_timesteps)
                            sample_global_ids = sample_global_ids[selected_mask]
                            sample_timesteps = sample_timesteps[selected_mask]
                            volume_batch_fulfilled = True

                        # Add this streamline's length to total length
                        total_volume_timesteps += np.sum(sample_timesteps)

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
