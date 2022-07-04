# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import Union

import numpy as np
import torch

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset

from scilpy.tracking.propagator import AbstractPropagator

from dwi_ml.models.main_models import MainModelAbstract, MainModelOneInput

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
    dataset: MultisubjectSubset

    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
                 model: MainModelAbstract, step_size: float, rk_order: int,
                 algo: str, theta: float, model_uses_streamlines: bool = False,
                 device=None):
        """
        Parameters
        ----------
        dataset: MultisubjectSubset
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
        # Dataset will be reloaded at sub-processes
        super().__init__(dataset, step_size, rk_order)

        if rk_order > 1:
            logger.warning("dwi_ml is not ready for runge-kutta integration."
                           "Changing to rk_order 1.")
            self.rk_order = 1

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

        # The model uses the streamline, so we need to keep track of it as
        # additional input. List of lines. All lines have the same number of
        # points, as they are being propagated together.
        # List[list[list]]: nb_lines x (nb_points, 3).
        self.current_lines = None  # type: Union[list, None]

        # Propagate method call will influence how we get the streamlines
        self._track_multiple_lines = None  # type: Union[bool, None]

    def move_to(self, device):
        #  Reminder. Contrary to tensors, model.to overwrites the model.
        self.model.move_to(device=device)
        self.device = device

    def reset_data(self, reload_data: bool = True):
        """
        Reset data before starting a new process during multi-processing.

        reload_data: bool
            If true, reload data to cache. Else, erase all data and hdf handles
            from memory.
        """
        if self.dataset.is_lazy:
            # Empty cache
            self.dataset.volume_cache_manager = None

            # Remove all handles
            self.dataset.close_all_handles()

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
        if isinstance(line[0], np.ndarray):  # List of coords = single tracking
            v_in = super().prepare_backward(line, forward_dir)

            # v_in is in double format (np.float64) but it looks like we need
            # float32.
            # todo From testing with Learn2track. Always true?
            if v_in is not None:
                v_in = v_in.astype(np.float32)
        else:  # simultaneous tracking.
            v_in = []
            for i in range(len(line)):
                this_v_in = super().prepare_backward(line[i],
                                                     forward_dir[i])
                if this_v_in is not None:
                    this_v_in = this_v_in.astype(np.float32)
                v_in.append(this_v_in)

        return v_in

    def multiple_lines_update(self, lines_that_continue: list):
        """
        This is used by the tracker with simulatenous tracking, in case you
        need to update your model's memory when removing a streamline.
        """
        pass

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
        self._track_multiple_lines = multiple_lines

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
            n_new_pos, n_new_dir, are_directions_valid = \
                self._propagate_multiple_lines(line, v_in)

            return n_new_pos, n_new_dir, are_directions_valid
        else:
            # We could use self._propagate_multiple_lines([line], v_in).
            # Keeping this version with super in case we prepare everything for
            # use with runge_kutta order > 1.
            new_pos, new_dir, is_direction_valid = \
                super().propagate(line, v_in)

            return new_pos, new_dir, is_direction_valid

    def _propagate_multiple_lines(self, lines, n_v_in):
        """
        Equivalent of self.propagate() for multiple lines. We do not call
        super's at it does not support multiple lines. Super's method supports
        rk order. We don't. We skip this and directly call
        _sample_next_direction_or_go_straight.
        """
        # Keeping one coordinate per streamline; the last one.
        # If model needs the streamlines, use current memory.
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

        n_v_out = self._sample_next_direction(n_pos, n_v_in)
        are_directions_valid = np.array([False if v_out is None else True
                                         for v_out in n_v_out])

        n_v_out = [n_v_out[i] if are_directions_valid[i] else n_v_in[i]
                   for i in range(len(n_v_in))]

        return are_directions_valid, n_v_out

    def _sample_next_direction(self, pos, v_in):
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
        if self._track_multiple_lines:
            n_pos = pos
            streamline_lengths = [1 for _ in n_pos]
        else:
            n_pos = [pos]
            streamline_lengths = None

        # Tracking field returns the model_outputs
        model_outputs = self._get_model_outputs_at_pos(n_pos)

        start_time = datetime.now()
        if self.algo == 'prob':
            next_dirs = self.model.sample_tracking_direction_prob(model_outputs)
        else:
            next_dirs = self.model.get_tracking_direction_det(model_outputs)
        duration_direction_getter = datetime.now() - start_time

        start_time = datetime.now()
        for i in range(len(n_pos)):
            next_dirs[i] /= np.linalg.norm(next_dirs[i])
            # Verify curvature, else return None.
            # toDo could we find a better solution for proba tracking?
            #  Resampling until angle < theta? Easy on the sphere (restrain
            #  probas on the sphere pics inside a cone theta) but what do we do
            #  for other models? Ex: For the Gaussian direction getter?
            if v_in[i] is not None:
                next_dirs[i] = self._verify_angle(next_dirs[i], v_in[i])
        duration_norm_angle = datetime.now() - start_time

        logging.debug("Time for direction getter: {}s. Time for normalization "
                      "+ verifying angle: {}s."
                      .format(duration_direction_getter.total_seconds(),
                              duration_norm_angle.total_seconds()))

        if not self._track_multiple_lines:
            return next_dirs[0]

        return next_dirs

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
            # Todo. This is not perfect yet. Sending data to new device at each
            #  new point. Could it already be a tensor in memory?
            lines = [torch.cat((torch.tensor(line),
                               torch.zeros(1, 3)), dim=0).to(self.device)
                     for line in self.current_lines]

            start_time = datetime.now()
            model_outputs = self.model(inputs, lines)
        else:
            start_time = datetime.now()
            model_outputs = self.model(inputs)
        duration_running_model = datetime.now() - start_time

        logger.debug("Time to run the model: {}"
                     .format(duration_running_model.total_seconds()))

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
        cos_angle = min(max(-1.0, float(cos_angle)), 1.0)

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
    def __init__(self, dataset: MultisubjectSubset, subj_idx: int,
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
        step_size: NOW IN VOXEL RESOLUTION.
        """
        super().__init__(dataset, subj_idx, model, step_size, rk_order, algo,
                         theta, model_uses_streamlines, device)

        # Find group index in the data
        self.volume_group = dataset.volume_groups.index(input_volume_group)

        # To help prepare the inputs
        if hasattr(model, 'neighborhood_points'):
            self.neighborhood_points = model.neighborhood_points
        else:
            self.neighborhood_points = None
        self.volume_group_str = input_volume_group

        if self.dataset.is_lazy and self.dataset.cache_size == 0:
            logger.warning("With lazy data and multiprocessing, you should "
                           "not keep cache size to zero. Data would be "
                           "loaded again at each propagation step!")

    def _prepare_inputs_at_pos(self, n_pos):
        """
        Prepare inputs at current position: get the volume and interpolate at
        current coordinate (possibly get the neighborhood coordinates too).

        Params
        ------
        pos: list[n x [[x, y, z]]]
            List of n "streamlines" composed of one point.
        """
        # torch trilinear interpolation uses origin='corner', space=vox.
        # We're ok.
        lines = [[point] for point in n_pos]

        inputs, _ = MainModelOneInput.prepare_batch_one_input(
            lines, self.dataset, self.subj_idx, self.volume_group,
            self.neighborhood_points, self.device)
        return inputs
