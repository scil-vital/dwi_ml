# -*- coding: utf-8 -*-
import os
from math import ceil
from typing import Any, Dict, List, Union

import h5py
import nibabel as nib
import numpy as np
import torch
import tqdm
from dipy.io.stateful_tractogram import Space
from dipy.io.streamline import load_tractogram
from dipy.tracking.streamline import length as slength
from nibabel.affines import apply_affine
from nibabel.streamlines import ArraySequence, Tractogram

from dwi_ml.data.dataset.single_subject_containers import (
    MRIDataVolume, SubjectData)
from dwi_ml.experiment.timer import Timer
from dwi_ml.tracking.step_tracker import (StepTracker,
                                          PreInitializedStepTracker)
from dwi_ml.tracking.utils import StoppingFlags, count_flags


class TrackerAbstract(object):
    """Use an existing model to track on a new subject."""

    def __init__(self, model: torch.nn.Module,
                 dataset_file: str, subject_id: str,
                 seeding_file: str, tracking_file: str = None,
                 rng_seed: int = 1234, n_seeds_per_voxel: int = 1,
                 seeding_ref: str = None, use_gpu: bool = True,
                 add_neighborhood: float = None,
                 add_previous_dir: bool = False):
        """
        Parameters
        ----------
        model: torch.nn.Module
            Trained model that will generate the tracking directions.
            MUST HAVE A sample_tracking_directions FUNCTION AND A eval FUNCTION.
        dataset_file : str
            Path to dataset file (.hdf5).
        subject_id : str
            Subject id to fetch from the dataset file.
        seeding_file : str
            Path to seeding mask (.nii.gz) or seeding streamlines (.tck|.trk).
        tracking_file : str (optional)
            Path to binary tracking mask (.nii.gz).
        rng_seed : int
            Random seed.
        n_seeds_per_voxel : int
            Number of random seeds to be initialized in each voxel of the
            seeding mask.
        seeding_ref : str
            Path to reference file neceserray if `seeding_file` is a tractogram.
        use_gpu : bool
            If False, do not use the GPU for tracking.
        add_neighborhood : float (optional)
            If given, add neighboring information to the input signal at the
            given distance in each axis (in mm).
        add_previous_dir : bool (optional)
            If given, add the streamline previous direction to the input signal.
        """
        self.rng = np.random.RandomState(seed=rng_seed)
        self.n_seeds_per_voxel = n_seeds_per_voxel
        self.use_gpu = use_gpu

        # Load subject
        with h5py.File(dataset_file, 'r') as hdf_file:
            assert subject_id in list(hdf_file.keys()), \
                "Subject {} not found in file: {}".format(subject_id,
                                                          dataset_file)

            self.tracto_data = SubjectData.create_from_hdf(hdf_file[subject_id])
            self.tracto_data.input_dv.subject_id = subject_id

        ext = os.path.splitext(seeding_file)[1]
        if ext in ['.nii', '.gz']:
            # Load seeding mask (should be a binary image)
            seeding_image = nib.load(seeding_file)
            self.seeding = seeding_image.get_fdata()
            self.affine_seedsvox2rasmm = seeding_image.affine
        elif ext in ['.tck', '.trk']:
            # Load seeding streamlines
            if seeding_ref is None:
                raise ValueError("A reference is necessary to load a "
                                 "tractogram; please use --seeding-ref")
            seeding_ref_img = nib.load(seeding_ref)
            seeding_tractogram = load_tractogram(seeding_file, seeding_ref_img,
                                                 to_space=Space.VOX)
            seeding_tractogram.to_center()
            self.seeding = seeding_tractogram.streamlines
            self.affine_seedsvox2rasmm = seeding_ref_img.affine

        # Load tracking mask if given
        self.tracking_dv = None
        if tracking_file:
            tracking_image = nib.load(tracking_file)
            self.tracking_dv = MRIDataVolume(
                data=tracking_image.get_fdata(dtype=np.float32),
                affine_vox2rasmm=tracking_image.affine)

        # Compute affine to bring seeds into DWI voxel space
        # affine_seedsvox2dwivox : seeds voxel space => rasmm space => dwi voxel space
        affine_rasmm2dwivox = np.linalg.inv(
            self.tracto_data.input_dv.affine_vox2rasmm)
        self.affine_seedsvox2dwivox = np.dot(
            affine_rasmm2dwivox, self.affine_seedsvox2rasmm)

        # Other parameters
        self.add_neighborhood = add_neighborhood
        self.add_previous_dir = add_previous_dir
        self.model = model
        self.model.eval()

    @staticmethod
    def _load_model(model_path: str, hyperparameters: Dict[str, Any]):
        raise NotImplementedError

    @staticmethod
    def _run_tracker(tracker: StepTracker, seeds: Union[np.ndarray, List]) \
            -> Tractogram:
        """Runs a tracker, starting from the provided seeds, and returns the
        final tractogram.

        Parameters
        ----------
        tracker : StepTracker
            Tracker that will grow streamlines
        seeds : np.ndarray with shape (n_streamlines, 3) or (n_streamlines,
            n_points, 3), or list of np.ndarray with shape (n_points, 3)
            Initial starting points or initial streamlines.

        Returns
        -------
        tractogram : nib.Tractogram
            Tractogram containing all streamlines and stopping information.
        """
        tractogram = None
        tracker.initialize(seeds)

        length_stopping_criterion = \
            tracker.stopping_criteria[StoppingFlags.STOPPING_LENGTH]
        with torch.no_grad(), \
             tqdm.tqdm(range(
                 length_stopping_criterion.keywords['max_nb_steps'])) as pbar:
            for _ in pbar:
                tracker.grow_step()

                if tractogram is None:
                    tractogram = tracker.harvest()
                else:
                    tractogram += tracker.harvest()

                if tracker.is_finished_tracking():
                    pbar.close()
                    break

        return tractogram

    @staticmethod
    def _get_tracking_seeds_from_mask(mask: np.ndarray,
                                      affine_seedsvox2dwivox: np.ndarray,
                                      n_seeds_per_voxel: int,
                                      rng: np.random.RandomState) -> np.ndarray:
        """Given a binary seeding mask, get seeds in DWI voxel space using the
        provided affine.

        Parameters
        ----------
        mask : np.ndarray with shape (X,Y,Z)
            Binary seeding mask.
        affine_seedsvox2dwivox : np.ndarray
            Affine to bring the seeds from their voxel space to the input voxel
             space.
        n_seeds_per_voxel : int
            Number of seeds to generate in each voxel
        rng : np.random.RandomState
            Random number generator

        Returns
        -------
        seeds : np.ndarray with shape (N_seeds, 3)
            Position of each initial tracking seeds
        """
        seeds = []
        indices = np.array(np.where(mask)).T
        for idx in indices:
            seeds_in_seeding_voxel = idx + rng.uniform(-0.5, 0.5,
                                                       size=(n_seeds_per_voxel, 3))
            seeds_in_dwi_voxel = nib.affines.apply_affine(affine_seedsvox2dwivox,
                                                          seeds_in_seeding_voxel)
            seeds.extend(seeds_in_dwi_voxel)
        seeds = np.array(seeds, dtype=np.float32)
        return seeds

    def track(self, max_length: float, batch_size: int = None,
              step_size: float = None, max_angle: float = None,
              min_length: float = None) -> Tractogram:
        """Track a whole tractogram from the seeds. First run forward,
        then backwards using the streamlines that were tracked.

        Parameters
        ----------
        max_length : float
            Maximum streamline length in mm.
        batch_size : int (optional)
            Number of streamlines that should be tracked at the same time.
            If None, try with a full batch and divide by 2 until it fits into
            memory.
        step_size : float (optional)
            Step size in mm. If None, use the model outputs without scaling.
        max_angle : float
            Maximum angle in degrees that two consecutive segments can have
            between each other (corresponds to the maximum half-cone angle).
        min_length : float
            Minimum streamline length in mm.
            (If given, streamlines shorter than this length will be discarded).

        Returns
        -------
        tractogram : nib.Tractogram
            Tractogram with all the tracked streamlines.
        """

        if isinstance(self.seeding, np.ndarray):
            # Get random seeds from seeding mask
            seeds = self._get_tracking_seeds_from_mask(
                self.seeding, self.affine_seedsvox2dwivox,
                self.n_seeds_per_voxel, self.rng)
        else:
            # Use streamlines as seeds
            seeds = self.seeding

        # Compute minimum length voxel-wise
        if min_length:
            min_length_vox = convert_mm2vox(
                min_length, self.tracto_data.input_dv.affine_vox2rasmm)

        # Initialize trackers
        if isinstance(seeds, (list, ArraySequence)):
            forward_tracker_cls = PreInitializedStepTracker
        else:
            forward_tracker_cls = StepTracker
        forward_step_tracker = forward_tracker_cls(
            model=self.model, input_dv=self.tracto_data.input_dv,
            mask_dv=self.tracking_dv, step_size=step_size,
            add_neighborhood=self.add_neighborhood,
            add_previous_dir=self.add_previous_dir, max_length=max_length,
            max_angle=max_angle, use_gpu=self.use_gpu)
        backwards_step_tracker = PreInitializedStepTracker(
            model=self.model, input_dv=self.tracto_data.input_dv,
            mask_dv=self.tracking_dv, step_size=step_size,
            add_neighborhood=self.add_neighborhood,
            add_previous_dir=self.add_previous_dir, max_length=max_length,
            max_angle=max_angle, use_gpu=self.use_gpu)

        if step_size:
            print("Tracking using a step size of {:.3f} mm "
                  "({:.3f} voxels)".format(step_size, forward_step_tracker.step_size_vox))
        else:
            print("Tracking using the model output without scaling")
        print("Tracking from {} seeds".format(len(seeds)))

        if batch_size is None:
            batch_size = len(seeds)

        # Try batch sizes until it fits into memory (divide by 1.25 if it
        # doesn't and try again)
        while True:
            print("Trying a batch size of {} streamlines".format(batch_size))
            n_iter = int(ceil(len(seeds) / batch_size))
            try:
                tractogram = None

                for i, start in enumerate(range(0, len(seeds), batch_size)):
                    end = start + batch_size
                    print("Iteration {} of {}".format(i + 1, n_iter))

                    # Forward tracking
                    with Timer("Forward pass", newline=True, color='green'):
                        batch_tractogram = self._run_tracker(forward_step_tracker,
                                                             seeds[start:end])

                    stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)

                    print("Forward pass stopped because of - mask: {:,}\t "
                          "curvature: {:,}\t length: {:,}".format(
                        count_flags(stopping_flags, StoppingFlags.STOPPING_MASK),
                        count_flags(stopping_flags, StoppingFlags.STOPPING_CURVATURE),
                        count_flags(stopping_flags, StoppingFlags.STOPPING_LENGTH)))

                    # Backwards tracking
                    # Flip streamlines to initialize backwards tracker
                    streamlines_init = [s[::-1] for s in batch_tractogram.streamlines]

                    with Timer("Backwards pass", newline=True, color='green'):
                        batch_tractogram = self._run_tracker(backwards_step_tracker,
                                                             streamlines_init)

                    stopping_flags = batch_tractogram.data_per_streamline['stopping_flags'].astype(np.uint8)

                    print("Backwards pass stopped because of - mask: {:,}\t "
                          "curvature: {:,}\t length: {:,}".format(
                        count_flags(stopping_flags, StoppingFlags.STOPPING_MASK),
                        count_flags(stopping_flags, StoppingFlags.STOPPING_CURVATURE),
                        count_flags(stopping_flags, StoppingFlags.STOPPING_LENGTH)))

                    # Filter short streamlines
                    if min_length:
                        lengths_vox = slength(batch_tractogram.streamlines)
                        to_keep = np.where(lengths_vox > min_length_vox)
                        print("Removing {} streamlines that were under {} mm".format(
                            len(batch_tractogram) - len(to_keep[0]), min_length))

                        # Make a copy because indexing an ArraySequence creates
                        # a "view" with the same _data property, which causes problems
                        # when extending tractograms
                        batch_tractogram = batch_tractogram[to_keep].copy()

                    if tractogram is None:
                        tractogram = batch_tractogram
                    else:
                        tractogram += batch_tractogram
                return tractogram

            except MemoryError:
                print("Not enough memory for a batch size of {} streamlines".format(batch_size))
                batch_size = int(batch_size / 1.25)
                if batch_size <= 0:
                    raise MemoryError("Not enough memory! You might need a "
                                      "bigger graphics card!")

            except RuntimeError as e:
                if "out of memory" in e.args[0] or "CuDNN error" in e.args[0]:
                    print("Not enough memory for a batch size of {} streamlines"
                          .format(batch_size))
                    batch_size = int(batch_size / 1.25)
                    if batch_size <= 0:
                        raise MemoryError("Not enough memory! You might need a "
                                          "bigger graphics card!")
                else:
                    raise e
