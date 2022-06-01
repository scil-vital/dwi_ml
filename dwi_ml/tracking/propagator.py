# -*- coding: utf-8 -*-
import logging

import numpy as np
import torch
from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset

from scilpy.tracking.propagator import AbstractPropagator

from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.models.main_models import MainModelAbstract

logger = logging.getLogger('tracker_logger')


class DWIMLPropagator(AbstractPropagator):
    """
    Abstract class for propagator object. Responsible for sampling the final
    direction through Runge-Kutta integration.

    Differences with traditional tractography (e.g. in scilpy):
        - current data values (ex, fODF, SH) are not known. The method needs to
        get the current data from the hdf5 and prepare it as an input with
        format expected by the model.
        - the output of the model is not necessarily a direction. It could be,
        for instance, the logits associated with the classes in a
        classification model. Or it could be the means and sigmas representing
        the learned gaussian. Each model should contain methods to sample a
        direction based on these outputs.
        - theta would be very complex to include here as a cone and will rather
        be used as stopping criteria, later.
    """
    def __init__(self, subset: MultisubjectSubset, subj_idx: int,
                 model: MainModelAbstract, step_size: float, rk_order: int,
                 algo: str, theta: float, model_uses_streamlines: bool = False,
                 device=None):
        """
        Parameters
        ----------
        subset: MultisubjectSubset
            Loaded testing set. Must be lazy to allow multiprocessing.
            Multi-subject tracking not implemented; it should contain only
            one subject.
        subj_idx: int
            Subject used for tracking.
        step_size: float
            The step size for tracking.
        rk_order: int
            Order for the Runge Kutta integration.
        algo: str
            'det' or 'prob'
        theta: float
            Maximum angle (radians) allowed between two steps during sampling
            of the next direction.
        model_uses_streamlines: bool
            If true, the current line in kept in memory to be added as
            additional input.
        """
        dataset = None  # Could load now but will be reloading at processes
        # initialization anyway.
        super().__init__(dataset, step_size, rk_order)

        if rk_order > 1:
            logger.warning("dwi_ml is not ready for runge-kutta integration."
                           "Changing to rk_order 1.")
            self.rk_order = 1

        self.subset = subset
        self.subj_idx = subj_idx
        self.model = model

        self.algo = algo
        if algo not in ['det', 'prob']:
            raise ValueError("Propagator's algo should be 'det' or 'prob'.")

        self.theta = theta

        self.device = device
        if device is not None:
            self.move_to(device)

        self.model_uses_streamlines = model_uses_streamlines
        if self.model_uses_streamlines:
            # Add a line parameter. The model uses the streamline, so we need
            # to keep track of it as additional input.
            self.line = []

    def move_to(self, device):
        #  Reminder. Contrary to tensors, model.to overwrites the model.
        self.model.to(device=device)
        self.device = device

    def reset_data(self, reload_data: bool = True):
        """
        Reset data before starting a new process during multi-processing.

        reload_data: bool
            If true, reload data to cache. Else, erase all data and hdf handles
            from memory.
        """
        if self.subset.is_lazy:
            # Empty cache
            self.subset.volume_cache_manager = None

            # Remove all handles
            for i in range(len(self.subset.subjs_data_list)):
                s = self.subset.subjs_data_list[i]
                s.hdf_handle = None

            if reload_data:
                self._load_subj_data()

    def _load_subj_data(self):
        # To be implemented in child classes
        raise NotImplementedError

    def prepare_forward(self, seeding_pos):
        """
        Prepare information necessary at the first point of the
        streamline for forward propagation: v_in and any other information
        necessary for the self.propagate method.

        Parameters
        ----------
        seeding_pos: tuple(x,y,z)

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
        """
        if self.model_uses_streamlines:
            # Reset line before starting a new streamline.
            self.line = [list(seeding_pos)]

        # Our models should be able to get a direction even from with no
        # information about previous inputs.
        return None

    def prepare_backward(self, line, forward_dir):
        """
        Preparing backward.

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed.
        forward_dir: ndarray (3,)
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,)
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        if self.model_uses_streamlines:
            self.line = line

        # Note. In our case, compared to scilpy, forward dir is None. So if the
        # forward tracking failed, we will just return None and try the
        # backward again with v_in=None. So basically, we will recompute
        # exactly the same model outputs as for the forward. But maybe the
        # sampling will create a new direction.
        v_in = super().prepare_backward(line, forward_dir)

        # v_in is in double format (np.float64) but it looks like we need
        # float32.
        # toDo From testing with Learn2track. Always true?
        if v_in is not None:
            return v_in.astype(np.float32)
        return v_in

    def propagate(self, pos, v_in):
        logger.debug("  Propagation step at pos {}".format(pos))
        new_pos, new_dir, is_direction_valid = super().propagate(pos, v_in)
        if self.model_uses_streamlines:
            self.line.append(list(new_pos))

        self._update_state_after_propagation_step(new_pos, new_dir)

        return new_pos, new_dir, is_direction_valid

    def _update_state_after_propagation_step(self, new_pos, new_dir):
        """
        Update memory state between propagation steps. Anything your model
        needs to deal with in memory during streamline generation (ex, memory
        of the previous directions).
        """
        pass

    def _sample_next_direction(self, pos, v_in):
        """
        Run the model to get the outputs, and sample a direction based on this
        information. None if the direction is more than theta angle
        from v_in. (Then, direction will be considered invalid in
        _sample_next_direction_or_go_straight and v_in will be followed
        instead, for a maximum or X invalid steps.)

        Parameters
        ----------
        pos: ndarray (3,)
            Current tracking position.
        v_in: ndarray (3,)
            Previous tracking direction.

        Return
        -------
        direction: ndarray (3,)
            A valid tracking direction. None if no valid direction is found.
            Direction should be normalized.
        """

        # Tracking field returns the model_outputs
        model_outputs = self._get_model_outputs_at_pos(pos)

        # Sampling a direction from this information.
        if self.algo == 'prob':
            next_dir = self.model.sample_tracking_direction_prob(model_outputs)
        else:
            next_dir = self.model.get_tracking_direction_det(model_outputs)

        # Normalizing
        next_dir /= np.linalg.norm(next_dir)

        # Verify curvature, else return None.
        # toDo could we find a better solution for proba tracking? Resampling
        #  until angle < theta? Easy on the sphere (restrain probas on the
        #  sphere pics inside a cone theta) but what do we do for other models?
        #  Ex: For the Gaussian direction getter?
        if v_in is not None and len(v_in) > 0:
            next_dir = self._verify_angle(next_dir, v_in)

        # Need to detach grad to be able to convert to numpy array
        return next_dir

    def _get_model_outputs_at_pos(self, pos):
        """
        Parameters
        ----------
        pos: ndarray (3,)
            Current position coordinates.
        """
        inputs = self._prepare_inputs_at_pos(pos)
        if self.model_uses_streamlines:
            # In case we are using runge-kutta integration
            # During training, we have one more point then the number of
            # inputs: the last point is only used to get the direction.
            # Adding a fake last point.
            line = torch.tensor(self.line + [[0., 0., 0.]])
            model_outputs = self.model([inputs], [line])
        else:
            model_outputs = self.model([inputs])

        return model_outputs

    def _prepare_inputs_at_pos(self, pos):
        raise NotImplementedError

    def _verify_angle(self, next_dir, v_in):
        """
        Note. With deterministic tracking on peaks, we would have to implement
        verification of the direction closest to v_in and possibly flip
        next_dir. Not done here, as next_dir comes from a machine learning
        algorithm which should already learn to output something in the right
        direction.
        """
        cos_angle = np.dot(next_dir / np.linalg.norm(next_dir),
                           v_in.T / np.linalg.norm(v_in))

        # Resolving numerical instabilities:
        cos_angle = min(max(-1.0, cos_angle), 1.0)

        angle = np.arccos(cos_angle)
        if angle > self.theta:
            return None
        return next_dir


class DWIMLPropagatorOneInput(DWIMLPropagator):
    """
    This version of the propagator is for when data is represented as
    x: one volume input (formatted through the model, possibly with
       neighborhood).

    The class is used to get outputs from the model.

    This is in fact similar to our batch sampler (with inputs): it needs to get
    the data points from the volume (+possibly add a neighborhood) and
    interpolate the data.
    """
    def __init__(self, subset: MultisubjectSubset, subj_idx: int,
                 model: MainModelAbstract, input_volume_group: str,
                 step_size: float, rk_order: int, algo: str, theta: float,
                 model_uses_streamlines: bool, device=None):
        """
        Additional params compared to super:
        ------------------------------------
        input_volume_group: str
            The volume group to use as input in the model.
        neighborhood_points: np.ndarray
            The list of neighborhood points (does not contain 0,0,0 point)
        """
        super().__init__(subset, subj_idx, model, step_size, rk_order, algo,
                         theta, model_uses_streamlines, device)

        # Find group index in the data
        self.volume_group = subset.volume_groups.index(input_volume_group)

        # To help prepare the inputs
        if hasattr(model, 'neighborhood_points'):
            self.neighborhood_points = model.neighborhood_points
        else:
            self.neighborhood_points = None
        self.volume_group_str = input_volume_group

    def _load_subj_data(self):
        # This will open a new handle and get the volume to cache.
        # If cache is not activated, tracking will load the data at each
        # propagation step!
        if self.subset.is_lazy and self.subset.cache_size == 0:
            raise ValueError("With lazy data and multiprocessing, you should "
                             "not keep cache size to zero. Data would be "
                             "loaded again at each propagation step!")
        else:
            # Load as DataVolume
            self.dataset = self.subset.get_volume_verify_cache(
                self.subj_idx, self.volume_group, as_tensor=False)

    def _prepare_inputs_at_pos(self, pos):
        # Dataset should already be loaded if multiprocessing. Else, load it
        # the first time we actually need it.
        if self.dataset is None:
            self._load_subj_data()

        # torch trilinear interpolation uses origin='corner', space=vox.
        pos_vox = self.dataset.voxmm_to_vox(*pos, self.origin)
        if self.origin == 'center':
            pos_vox += 0.5

        # Adding dim. array(3,) should become array (1,3)
        pos_vox = np.expand_dims(pos_vox, axis=0)

        # Get data as tensor. The MRI data should already be in the cache.
        dataset_as_tensor = self.subset.get_volume_verify_cache(
            self.subj_idx, self.volume_group, as_tensor=True)

        # Same as in the batch sampler:
        # Prepare the volume data, possibly adding neighborhood
        subj_x_data, _ = interpolate_volume_in_neighborhood(
            dataset_as_tensor, pos_vox, self.neighborhood_points,
            self.device)

        # Return inputs as a tuple containing all inputs. The comma is
        # important.
        return subj_x_data
