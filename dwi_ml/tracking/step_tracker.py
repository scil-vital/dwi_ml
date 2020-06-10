# -*- coding: utf-8 -*-
import functools
from typing import List, Tuple

import nibabel as nib
from nibabel.streamlines import Tractogram
import numpy as np
import torch

from dwi_ml.data.dataset.single_subject_containers import MRIDataVolume
from dwi_ml.data.processing.volume.interpolation import (
    torch_trilinear_interpolation)
from dwi_ml.tracking.utils import (StoppingFlags, filter_stopping_streamlines,
                                   is_outside_mask, is_too_curvy, is_too_long)


class StepTracker(object):
    """ Generate streamlines using a pretrained recurrent model.                                                # A comprendre si ça doit être a sequence model
        Streamlines will be stopped according to predefined criteria. """

    def __init__(self, model: torch.nn.Module, input_dv: MRIDataVolume,
                 max_length: float, mask_dv: MRIDataVolume = None,
                 step_size: float = None, add_neighborhood: float = None,
                 add_previous_dir: bool = False,
                 max_angle: float = None, use_gpu=True):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Trained model that will generate the tracking directions.
            MUST HAVE A sample_tracking_directions FUNCTION.
        input_dv : MRIDataVolume
            Input information used by the model for tracking.
        max_length : float
            Maximum streamline length in mm.
        mask_dv : MRIDataVolume
            Mask that defines where streamlines are allowed to track.
            If None, streamlines are allowed to go everywhere.
        step_size : float (optional)
            Desired step size in mm. If None, use the model outputs without
            scaling.
        add_neighborhood : float (optional)
            If given, add neighboring information to the input signal at the
            given distance in each axis (in mm).
        add_previous_dir : bool (optional)
            If given, add the streamline previous direction to the input signal.
        max_angle : float (optional)
            Maximum angle in degrees that two consecutive segments can have
            between each other (corresponds to the maximum half-cone angle).
        use_gpu : bool
            Use the GPU; if False, use CPU.
        """
        self.model = model
        self.input_dv = input_dv
        self.mask_dv = mask_dv
        self.use_gpu = use_gpu

        self.data_tensor = torch.as_tensor(self.input_dv.data)

        # Set device
        self.device = None
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        # Stopping criteria is a dictionary that maps `StoppingFlags` to
        # functions that indicate whether streamlines should stop or not.
        self.stopping_criteria = {}

        # Add mask stopping criterion
        if self.mask_dv is None:
            mask_data = np.ones(self.input_dv.shape[:3])
            affine_dwivox2maskvox = np.eye(4)
        else:
            mask_data = mask_dv.data.numpy()
            # Compute the affine to align dwi voxel coordinates with mask voxel
            # coordinates
            # affine_dwivox2maskvox : dwi voxel space => rasmm space => mask voxel space
            affine_rasmm2maskvox = np.linalg.inv(mask_dv.affine_vox2rasmm)
            affine_dwivox2maskvox = np.dot(affine_rasmm2maskvox,
                                           input_dv.affine_vox2rasmm)
        self.stopping_criteria[StoppingFlags.STOPPING_MASK] = \
            functools.partial(is_outside_mask, mask=mask_data,
                              affine_vox2mask=affine_dwivox2maskvox,
                              threshold=0.5)

        # Set maximum length
        if step_size:
            max_nb_steps = int(np.ceil(max_length / step_size))
            self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
                functools.partial(is_too_long, max_nb_steps=max_nb_steps)
        else:

            max_length_vox = convert_mm2vox(max_length,
                                            input_dv.affine_vox2rasmm)
            # Define max_nb_steps to make sure model doesn't run indefinitely
            # with really small steps
            max_nb_steps = 1000
            self.stopping_criteria[StoppingFlags.STOPPING_LENGTH] = \
                functools.partial(is_too_long,
                                  max_nb_steps=max_nb_steps,
                                  max_length=max_length_vox)

        # Set maximum angle if given
        if max_angle is not None:
            self.stopping_criteria[StoppingFlags.STOPPING_CURVATURE] = \
                functools.partial(is_too_curvy, max_theta=max_angle)

        # Convert step size from rasmm to voxel space
        self.step_size_vox = None
        if step_size:
            self.step_size_vox = convert_mm2vox(step_size,
                                                input_dv.affine_vox2rasmm)

        # Convert neighborhood to voxel space
        self.add_neighborhood_vox = None
        if add_neighborhood:
            self.add_neighborhood_vox = convert_mm2vox(add_neighborhood,
                                                       input_dv.affine_vox2rasmm)
            self.neighborhood_directions = get_interp_neighborhood_vectors(
                radius=self.add_neighborhood_vox)

        self.add_previous_dir = add_previous_dir

        self.streamlines = None
        self.model_states = None

    def initialize(self, seeds: np.ndarray):
        """Initialize tracking seeds and model, then send data on the chosen
        device.

        Parameters
        ----------
        seeds : np.ndarray with shape (n_streamlines, 3) or
                                      (n_streamlines, n_points, 3)
            Initial starting points for streamlines.
        """
        if seeds.ndim == 2:
            seeds = seeds[:, None, :]
        self.streamlines = seeds.copy()

        self.model_states = None

        # Move data to device (NOTE: Only the model is modified in-place)
        self.model.to(device=self.device)
        self.data_tensor = self.data_tensor.to(device=self.device)

    def grow_step(self, states: Tuple[torch.Tensor, ...],
                  step_size: float):
        """ Grow streamlines for one step forward """
        # Track only from endpoints using provided model states
        coords = self.streamlines[:, -1, :]

        # Get input data
        if self.add_neighborhood_vox:
            # Extend the coords array with the neighborhood coordinates
            n_coords = coords.shape[0]
            coords = np.repeat(coords, self.neighborhood_directions.shape[0],
                               axis=0)
            coords[:, :3] += np.tile(self.neighborhood_directions,
                                     (n_coords, 1))

            coords_tensor = torch.as_tensor(coords, device=self.device)

            # Evaluate signal as if all coords were independent
            signal_tensor = torch_trilinear_interpolation(self.data_tensor,
                                                          coords_tensor)

            # Reshape signal into (n_coords, new_feature_size)
            new_feature_size = \
                self.input_dv.shape[-1] * self.neighborhood_directions.shape[0]
            signal_tensor = signal_tensor.reshape(n_coords, new_feature_size)
        else:
            coords_tensor = torch.as_tensor(coords, device=self.device)
            signal_tensor = torch_trilinear_interpolation(self.data_tensor,
                                                          coords_tensor)

        if self.add_previous_dir:
            if self.streamlines.shape[1] == 1:
                previous_dir = torch.zeros((signal_tensor.shape[0], 3),
                                           dtype=torch.float32,
                                           device=self.device)
            else:
                previous_dir = torch.as_tensor(
                    self.streamlines[:, -1, :] - self.streamlines[:, -2, :],
                    dtype=torch.float32, device=self.device)
            signal_tensor = torch.cat((signal_tensor, previous_dir), dim=1)

        # Run model on data
        with torch.no_grad():
            torch_directions, new_states = \
                self.model.sample_tracking_directions(signal_tensor,
                                                      self.model_states)
            directions = torch_directions.cpu().numpy()

        if self.step_size_vox:
            # Scale directions to step size
            normalized_directions = directions / np.sqrt(np.sum(directions ** 2,
                                                                axis=-1,
                                                                keepdims=True))
            directions = normalized_directions * self.step_size_vox

        # Grow streamlines one step forward
        new_streamlines = np.concatenate(
            [self.streamlines,
             self.streamlines[:, [-1], :] + directions[:, None, :]],
            axis=1)

        self.streamlines = new_streamlines
        self.model_states = new_states

    def is_stopping(self, streamlines: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check which streamlines should stop or not according to the
        predefined stopping criteria

        Parameters
        ----------
        streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
            Streamlines that will be checked.

        Returns
        -------
        continue_idx : np.ndarray
            Indices of the streamlines that should continue.
        stopping_idx : np.ndarray
            Indices of the streamlines that should stop.
        stopping_flags : np.ndarray
            `StoppingFlags` that triggered stopping for each stopping streamline.
        """
        continue_idx, stopping_idx, stopping_flags = \
            filter_stopping_streamlines(streamlines, self.stopping_criteria)
        return continue_idx, stopping_idx, stopping_flags

    def is_finished_tracking(self):
        """Returns True if all streamlines have stopped
        (i.e. if they have all triggered a stopping flag)

        Returns
        -------
        is_finished_tracking : bool
            True if there are any remaining streamlines, False otherwise.
        """
        return len(self.streamlines) == 0

    def get_from_flag(self, flag: StoppingFlags):
        """Get streamlines that stopped only for a given stopping flag

        Parameters
        ----------
        flag : tracking_utils.StoppingFlags
            Flag to match to stopped streamlines

        Returns
        -------
        stopping_idx : np.ndarray
            The indices corresponding to the streamlines stopped by the provided
            flag.
        """
        _, stopping_idx, stopping_flags = self.is_stopping(self.streamlines)
        return stopping_idx[(stopping_flags & flag) != 0]

    def _keep(self, idx: np.ndarray):
        """Keep only streamlines corresponding to the given indices,
        and remove all others. The model states will be updated accordingly.

        Parameters
        ----------
        idx : np.ndarray
            Indices of the streamlines to keep.
        """
        self.streamlines = self.streamlines[idx]

        if isinstance(self.model_states, tuple):
            self.model_states = tuple(s[:, idx] for s in self.model_states)
        elif isinstance(self.model_states, torch.Tensor):
            self.model_states = self.model_states[:, idx]
        else:
            raise TypeError

    def harvest(self) -> Tractogram:
        """Internally keep only the streamlines that haven't stopped yet,
        and return the streamlines that triggered a stopping flag.

        Returns
        -------
        tractogram : nib.streamlines.Tractogram
            Tractogram containing the streamlines that stopped tracking,
            along with the stopping_flags information in
            `tractogram.data_per_streamline`
        """
        continue_idx, stopping_idx, stopping_flags = \
            self.is_stopping(self.streamlines)

        # Drop last point since it triggered the flag
        stopped_streamlines = self.streamlines[stopping_idx, :-1]
        tractogram = Tractogram(
            streamlines=stopped_streamlines,
            data_per_streamline={"stopping_flags": stopping_flags})

        # Keep only streamlines that should continue
        self._keep(continue_idx)

        return tractogram


class PreInitializedStepTracker(StepTracker):
    """Subclass of `Tracker` used to track from preexisting streamlines.
    The model still needs to iterate over all the streamlines steps to update
    its states.
    """

    def __init__(self, model: torch.nn.Module, input_dv: MRIDataVolume,
                 max_length: float, mask_dv: MRIDataVolume = None,
                 step_size: float = None, add_neighborhood: float = None,
                 add_previous_dir: bool = False,
                 max_angle: float = None, use_gpu=True,
                 seeding_streamlines: List[np.ndarray] = None):
        """Initialize tracking seeds and model, then send data on the chosen
        device.

        Parameters
        ----------
        seeding_streamlines : list of np.ndarray with shape (n_points, 3)
            Initial streamlines.
        """
        super().__init__(model, input_dv, max_length, mask_dv, step_size,
                         add_neighborhood, add_previous_dir, max_angle, use_gpu)

        # Added None default for seeding but just to keep super's init order for
        # the first arguments to avoid user's error.
        if seeding_streamlines is None:
            raise ValueError
        self.seeding_streamlines = seeding_streamlines

        # Number if initialization steps for each streamline
        self.n_init_steps = np.asarray(list(map(len, seeding_streamlines)))

        # Get the first point of each seed as the start of the new streamlines
        self.streamlines = np.asarray([s[0] for s in
                                       seeding_streamlines])[:, None, :]

        self.model_states = None

        # Move data to device
        self.model.to(device=self.device)
        self.data_tensor = self.data_tensor.to(device=self.device)

    def is_stopping(self, streamlines: np.ndarray) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Check which streamlines should stop or not according to the
        predefined stopping criteria.

        Streamlines that have not finished initilization will never be marked
        as `stopping`.

        Parameters
        ----------
        streamlines : np.ndarray with shape (n_streamlines, n_points, 3)
            Streamlines that will be checked.

        Returns
        -------
        continue_idx : np.ndarray
            Indices of the streamlines that should continue.
        stopping_idx : np.ndarray
            Indices of the streamlines that should stop.
        stopping_flags : np.ndarray
            `StoppingFlags` that triggered stopping for each stopping streamline
        """
        continue_idx, stopping_idx, stopping_flags = \
            super().is_stopping(streamlines)

        # Indices fo streamlines that are still being initialized
        is_still_initializing = self.n_init_steps >= self.streamlines.shape[1]

        # Streamlines that haven't finished initializing should keep going
        continue_idx = np.concatenate([continue_idx,
                                       [idx for idx in stopping_idx if
                                        is_still_initializing[idx]]])
        continue_idx = continue_idx.astype(int)

        # Streamlines that haven't finished initializing should not stop
        is_really_stopping = np.logical_not(is_still_initializing[stopping_idx])
        stopping_idx = stopping_idx[is_really_stopping]
        stopping_flags = stopping_flags[is_really_stopping]

        return continue_idx, stopping_idx, stopping_flags

    def grow_step(self):
        """Grow stored streamlines one step forward and update model states """
        super().grow_step()

        # Do not update streamlines that are still initializing
        predicted_streamlines = self.streamlines
        is_still_initializing = \
            self.n_init_steps >= predicted_streamlines.shape[1]
        if np.any(is_still_initializing):
            # Replace the last point of the predicted streamlines
            # with the seeding streamlines at the same position
            predicted_streamlines[is_still_initializing, -1] = [
                s[predicted_streamlines.shape[1] - 1] for s, still_init
                in zip(self.seeding_streamlines, is_still_initializing)
                if still_init]

        self.streamlines = predicted_streamlines

    def _keep(self, idx: np.ndarray):
        """Keep only streamlines corresponding to the given indices, and remove
        all others. The model states will be updated accordingly.

        Parameters
        ----------
        idx : np.ndarray
            Indices of the streamlines to keep.
        """
        super()._keep(idx)
        self.seeding_streamlines = [self.seeding_streamlines[i] for i in idx]
        self.n_init_steps = self.n_init_steps[idx]
