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
                 device=None, simultaneous_tracking: bool = False):
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

        self.simultanous_tracking = simultaneous_tracking
        self.device = device
        if device is not None:
            self.move_to(device)

        self.model_uses_streamlines = model_uses_streamlines
        # The model uses the streamline, so we need to keep track of it as
        # additional input. List of lines. All lines have the same number of
        # points, as they are being propagated together.
        # List[list[list]]: nb_lines x (nb_points, 3).
        self.current_lines = None  # type: list

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
            self.subset.close_all_handles()

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
        seeding_pos: tuple(x,y,z) or List[tuples]
            The 3D coordinates or, for simultaneous tracking, list of 3D
            coordinates.

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
        """
        # Our models should be able to get an initial direction even with no
        # information about previous inputs.
        if isinstance(seeding_pos, tuple):
            return None
        else:
            return [None for _ in seeding_pos]

    def prepare_backward(self, line, forward_dir):
        """
        Preparing backward.

        Parameters
        ----------
        line: List
            Result from the forward tracking, reversed. Single line: list of
            coordinates. Simulatenous tracking: list of list of coordinates.
        forward_dir: ndarray (3,) or List[ndarray]
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,) or list[ndarray]
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        # Note. In our case, compared to scilpy, forward dir is None. So if the
        # forward tracking failed, we will just return None and try the
        # backward again with v_in=None. So basically, we will recompute
        # exactly the same model outputs as for the forward. But maybe the
        # sampling will create a new direction.
        if self.simultanous_tracking:
            v_in = []
            for i in range(len(line)):
                this_v_in = super().prepare_backward(line[i],
                                                     forward_dir[i])

                # See comment below
                if this_v_in is not None:
                    v_in.append(this_v_in.astype(np.float32))
        else:
            v_in = super().prepare_backward(line, forward_dir)

            # v_in is in double format (np.float64) but it looks like we need
            # float32.
            # toDo From testing with Learn2track. Always true?
            if v_in is not None:
                v_in = v_in.astype(np.float32)
        return v_in

    def propagate(self, line, v_in, multiple_lines=False):
        """
        Params
        ------
        line: list[ndarrray (3,)] or list[list]
            Current line, or current lines if multiple_lines.
        v_in: ndarray (3,) or List[TrackingDirection]
            Previous tracking direction, (or list of if multiple_lines.)

        Return
        ------
        new_pos: ndarray (3,) or list
            The new segment position.
        new_dir: ndarray (3,) or list
            The new segment direction.
        is_direction_valid: bool or list
            True if new_dir is valid.
        """
        logger.debug("  Propagation step !")
        if self.model_uses_streamlines:
            # super() won't use the whole line as argument during the sampling
            # of next direction, but we need it. Add it in memory here.
            # To use with r_k order > 1, we would need to compute the
            # intermediate lines at each sub-computation, easier if we actually
            # send the streamline, we would need to change scilpy.
            if multiple_lines:
                self.current_lines = line
            else:
                self.current_lines = [line]

        if multiple_lines:
            new_pos, new_dir, is_direction_valid = \
                self._propagate_multiple_lines(line, v_in)
        else:
            new_pos, new_dir, is_direction_valid = \
                super().propagate(line, v_in)

        # Reset memory
        self.current_lines = None

        logging.warning("GOT NEW POS: {}, NEW DIR: {}, VALID?: {}"
                        .format(new_pos, new_dir, is_direction_valid))

        return new_pos, new_dir, is_direction_valid

    def _propagate_multiple_lines(self, lines, n_v_in):
        """
        Equivalent of self.propagate() for multiple lines. We do not call
        super's at it does not support multiple lines. Super's method supports
        rk order. We don't. We skip this and directly call
        _sample_next_direction_or_go_straight.
        """
        # Finding last coordinate
        n_pos = [line[-1] for line in lines]

        assert self.rk_order == 1

        are_directions_valid, n_new_dir = \
            self._sample_next_multiple_directions_or_go_straight(n_pos, n_v_in)

        n_new_pos = [n_pos[i] + self.step_size * np.array(n_new_dir[i])
                     for i in range(len(lines))]

        return n_new_pos, n_new_dir, are_directions_valid

    def _sample_next_multiple_directions_or_go_straight(self, n_pos, n_v_in):
        """
        Equivalent of super's _sample_next_direction_or_go_straight.
        """
        n_v_out = self._sample_next_direction(n_pos, n_v_in,
                                              multiple_lines=True)
        are_directions_valid = np.array([False if v_out is None else True
                                         for v_out in n_v_out])
        n_v_out[~are_directions_valid] = n_v_in[~are_directions_valid]

        return are_directions_valid, n_v_out

    def _sample_next_direction(self, pos, v_in, multiple_lines=False):
        """
        Run the model to get the outputs, and sample a direction based on this
        information. None if the direction is more than theta angle
        from v_in. (Then, direction will be considered invalid in
        _sample_next_direction_or_go_straight and v_in will be followed
        instead, for a maximum or X invalid steps.)

        Parameters
        ----------
        pos: ndarray (3,) or list of ndarrays
            Current tracking position(s).
        v_in: ndarray (3,) ir list of ndarrays
            Previous tracking direction(s).

        Return
        -------
        direction: ndarray (3,) or list of ndarrays
            Valid normalized tracking direction(s). None if no valid direction
            is found.
        """
        single_streamline = False
        if multiple_lines:
            n_pos = pos  # Multiple values were sent.
            n_v_in = v_in
        else:
            logging.warning("SINGLE STREAMLINE!")
            single_streamline = True
            n_pos = [pos]
            n_v_in = [v_in]

        nb_streamlines = len(n_pos)
        logging.warning("n_pos: {}".format(n_pos))
        logging.warning("n_v_in: {}".format(n_v_in))

        # Tracking field returns the model_outputs
        model_outputs = self._get_model_outputs_at_pos(n_pos)

        logging.warning("MODEL OUTPUTS {}".format(model_outputs))

        # Sampling a direction from this information.
        next_dirs = []
        for i in range(nb_streamlines):
            if self.algo == 'prob':
                next_dir = self.model.sample_tracking_direction_prob(
                    model_outputs[i])
            else:
                next_dir = self.model.get_tracking_direction_det(
                    model_outputs[i])

            logging.warning("NEXT DIR: {}".format(next_dir))
            # Normalizing
            next_dir /= np.linalg.norm(next_dir)

            # Verify curvature, else return None.
            # toDo could we find a better solution for proba tracking? Resampling
            #  until angle < theta? Easy on the sphere (restrain probas on the
            #  sphere pics inside a cone theta) but what do we do for other models?
            #  Ex: For the Gaussian direction getter?
            if n_v_in[i] is not None:
                next_dir = self._verify_angle(next_dir, n_v_in[i])

            logging.warning("streamline {} final next dir {}".format(i, next_dir))
            if single_streamline:
                return next_dir
            else:
                return next_dirs.append(next_dir)

    def _get_model_outputs_at_pos(self, n_pos):
        """
        Parameters
        ----------
        n_pos: list of ndarrays
            Current position coordinates for each streamline.
        """
        inputs = self._prepare_inputs_at_pos(n_pos)

        if self.model_uses_streamlines:
            # Verify that we have updated memory correctly.
            # for each line:
            # assert np.array_equal(pos, self.current_lines[-1])

            # During training, we have one more point then the number of
            # inputs: the last point is only used to get the direction.
            # Adding a fake last point.
            lines = [torch.cat((torch.tensor(line),
                               torch.zeros(1, 3)), dim=0)
                     for line in self.current_lines]

            logging.warning("INPUTS: {}".format(inputs))
            logging.warning("LINES: {}".format(lines))
            model_outputs = self.model(inputs, lines)
        else:
            model_outputs = self.model(inputs)

        logging.warning("-- outputs {}".format(model_outputs))
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
        logging.warning("?? {}".format(next_dir))
        logging.warning("?? {}".format(v_in))
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
            logger.warning("With lazy data and multiprocessing, you should "
                           "not keep cache size to zero. Data would be "
                           "loaded again at each propagation step!")

        # Load as DataVolume
        # toDo. Not good for GPU. But allows us access to voxmm_to_vox
        #  method.
        self.dataset = self.subset.get_volume_verify_cache(
            self.subj_idx, self.volume_group, as_tensor=False)

    def _prepare_inputs_at_pos(self, n_pos):
        """
        Prepare inputs at current position: get the volume and interpolate at
        current coordinate (possibly get the neighborhood coordinates too).

        Params
        ------
        pos: list of ndarrrays
            Current position (or list of positions for multiple tracking).
        """
        # Dataset should already be loaded if multiprocessing. Else, load it
        # the first time we actually need it.
        if self.dataset is None:
            self._load_subj_data()

        all_inputs = []
        for pos in n_pos:
            # torch trilinear interpolation uses origin='corner', space=vox.
            pos_vox = self.dataset.voxmm_to_vox(*pos, self.origin)
            if self.origin == 'center':
                pos_vox += 0.5

            # Adding dim. array(3,) should become array (1,3)
            pos_vox = np.expand_dims(pos_vox, axis=0)

            # Get data as tensor. The MRI data should already be in the cache.
            dataset_as_tensor = self.subset.get_volume_verify_cache(
                self.subj_idx, self.volume_group, as_tensor=True,
                device=self.device)

            # Same as in the batch sampler:
            # Prepare the volume data, possibly adding neighborhood
            subj_x_data, _ = interpolate_volume_in_neighborhood(
                dataset_as_tensor, pos_vox, self.neighborhood_points,
                self.device)

            all_inputs.append(subj_x_data)

        return all_inputs
