# -*- coding: utf-8 -*-
import logging
from datetime import datetime
from typing import Union, List

from dipy.io.stateful_tractogram import Space, Origin
import numpy as np
import torch

from scilpy.tracking.propagator import AbstractPropagator

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.models.main_models import ModelForTracking, MainModelOneInput

logger = logging.getLogger('tracker_logger')


class DWIMLPropagator(AbstractPropagator):
    """
    Abstract class for propagator object. Responsible for sampling the final
    direction (by running the model).

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
                 model: ModelForTracking, step_size: float,
                 algo: str, theta: float, verify_opposite_direction: bool,
                 device=None, normalize_directions: bool = True):
        """
        Parameters
        ----------
        dataset: MultisubjectSubset
            Loaded testing set. Must be lazy to allow multiprocessing.
            Multi-subject tracking not implemented; it should contain only
            one subject.
        subj_idx: int
            Subject used for tracking.
        model: ModelForTracking
            Your torch model.
        step_size: float
            The step size for tracking, in voxel space.
        algo: str
            'det' or 'prob'
        theta: float
            Maximum angle (radians) allowed between two steps during sampling
            of the next direction.
        verify_opposite_direction: bool
            If true, during propagation, inverse the direction if opposite
            direction is better aligned (i.e. data model is symmetrical).
            If false, your model should learn to guess the correct direction
            as output.
        device: torch device
            CPU or GPU
        normalize_directions: bool
            If true, normalize directions.
        """
        # Dataset will be managed differently. Not a DataVolume.
        # torch trilinear interpolation uses origin='corner', space=vox.
        super().__init__(dataset, step_size, rk_order=1,
                         space=Space.VOX, origin=Origin('corner'))

        self.subj_idx = subj_idx
        self.model = model

        self.algo = algo
        if algo not in ['det', 'prob']:
            raise ValueError("Propagator's algo should be 'det' or 'prob'.")

        self.theta = theta
        self.normalize_directions = normalize_directions
        if not normalize_directions and step_size != 1:
            logging.warning("Tracker not normalizing directions obtained as "
                            "output from the model. Using a step size other "
                            "than 1 does not really make sense. You probably "
                            "want to advance of exactly 1 * output.")
            
        # If output is already normalized, no need to do it again.
        if self.model.normalize_outputs:
            self.normalize_directions = False

        self.device = device
        if device is not None:
            self.move_to(device)

        # Contrary to super: normalize direction is optional
        self.normalize_directions = normalize_directions
        self.verify_opposite_direction = verify_opposite_direction

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
        seeding_pos: Tensor or List(Tensor)
            The 3D coordinates or, for simultaneous tracking, list of 3D
            coordinates.

        Returns
        -------
        tracking_info: Any
            Any tracking information necessary for the propagation.
        """
        # Our models should be able to get an initial direction even with no
        # starting information. Override if your model is different.
        return [None for _ in seeding_pos]

    def prepare_backward(self, lines, forward_dir):
        """
        Preparing backward.

        Parameters
        ----------
        lines: List[Tensor]
            Result from the forward tracking, reversed. Each tensor is of
            shape (nb_points, 3).
        forward_dir: List[Tensor]
            v_in chosen at the forward step.

        Returns
        -------
        v_in: ndarray (3,) or list[ndarray]
            Last direction of the streamline. If the streamline contains
            only the seeding point (forward tracking failed), simply inverse
            the forward direction.
        """
        # If forward tracking succeeded, takes the last direction, inverted and
        # normalized (if self.normalize_directions).

        # Note. In our case, compared to scilpy, forward dir is None. So if the
        # forward tracking failed, we will just return None and try the
        # backward again with v_in=None. So basically, we will recompute
        # exactly the same model outputs as for the forward. But maybe the
        # sampling will create a new direction.
        v_in = [None] * len(lines)
        for i in range(len(lines)):
            if len(lines[i]) > 1:
                v_in[i] = lines[i][-1, :] - lines[i][-2, :]
                if self.normalize_directions:
                    v_in[i] /= torch.linalg.norm(v_in[i])
            elif forward_dir[i] is not None:
                v_in[i] = torch.flip(forward_dir[i], (0,))

        return v_in

    def multiple_lines_update(self, lines_that_continue: list):
        """
        This is used by the tracker with simulatenous tracking, in case you
        need to update your model's memory when removing a streamline.
        """
        pass

    def propagate(self, lines, n_v_in):
        """
        Overriding super()'s propagate. We don't support rk orders.
        We skip this and directly call _sample_next_direction_or_go_straight.
        Also, in our case, sub-methods return tensors.

        Params
        ------
        line: List[Tensor]
            For each streamline, tensor of shape (nb_points, 3).
        n_v_in: list[Tensor(3,)]
            Previous tracking directions.

        Return
        ------
        n_new_pos: list[Tensor(3,)]
            The new segment position.
        n_new_dir: list[Tensor(3,)]
            The new segment direction.
        valid_dirs: Tensor(nb_streamlines,)
            True if new_dir is valid.
        """
        # Keeping one coordinate per streamline; the last one.
        n_pos = [line[-1, :] for line in lines]

        # Converting n_v_in to the same shape, [nb_streamlines, 3]
        empty_pos = torch.full((3,), fill_value=torch.nan, device=self.device)
        n_v_in = [v if v is not None else empty_pos for v in n_v_in]
        n_v_in = torch.vstack(n_v_in)
        if len(n_v_in.shape) == 1:
            n_v_in = n_v_in[None, :]

        # Equivalent of sample_next_direction_or_go_straight, but avoiding
        # loop.
        n_v_out = self._sample_next_direction(n_pos, n_v_in)
        valid_dirs = n_v_out[:, 0] != torch.nan
        n_v_out[~valid_dirs, :] = n_v_in[~valid_dirs, :]

        n_new_pos = [n_pos[i] + self.step_size * n_v_out[i] for i in
                     range(len(n_pos))]

        return n_new_pos, n_v_out, valid_dirs

    def _sample_next_direction(self, n_pos, n_v_in):
        """
        Run the model to get the outputs, and sample a direction based on this
        information. None if the direction is more than theta angle
        from v_in. (Then, direction will be considered invalid in
        _sample_next_direction_or_go_straight and v_in will be followed
        instead, for a maximum or X invalid steps.)

        Parameters
        ----------
        n_pos: List[Tensor]
            Current tracking position(s). Tensors are of shape (3,).
        n_v_in: tensor(nb_streamlines, 3)
            Previous tracking direction(s).

        Return
        -------
        next_dirs: tensor(nb_streamlines, 3)
            Valid tracking direction(s). None if no valid direction
            is found. If self.normalize_directions, direction must be
            normalized.
        """
        # Using the forward method to get the outputs.
        model_outputs = self._get_model_outputs_at_pos(n_pos)

        start_time = datetime.now()
        # Input to model is a list of streamlines but output should be
        # next_dirs = a tensor of shape [nb_streamlines, 3].
        next_dirs = self.model.get_tracking_directions(model_outputs, self.algo)

        duration_direction_getter = datetime.now() - start_time
        logging.debug("Time for direction getter: {}s."
                      .format(duration_direction_getter.total_seconds()))

        start_time = datetime.now()
        if self.normalize_directions:
            # Will divide along correct axis.
            norm = torch.linalg.norm(next_dirs, dim=-1)[:, None]
            next_dirs = torch.div(next_dirs, norm)
        next_dirs = self._verify_angle(next_dirs, n_v_in)
        duration_norm_angle = datetime.now() - start_time

        logging.debug("Time for normalization + verifying angle: {}s."
                      .format(duration_norm_angle.total_seconds()))

        return next_dirs

    def _get_model_outputs_at_pos(self, n_pos):
        """
        Parameters
        ----------
        n_pos: List[Tensor(1,3)]
            Current position coordinates for each streamline.
        """
        inputs = self._prepare_inputs_at_pos(n_pos)

        start_time = datetime.now()
        model_outputs = self.model(inputs)
        duration_running_model = datetime.now() - start_time

        logger.debug("Time to run the model: {}"
                     .format(duration_running_model.total_seconds()))

        return model_outputs

    def _prepare_inputs_at_pos(self, pos):
        raise NotImplementedError

    def _verify_angle(self, next_dirs: torch.Tensor, n_v_in: torch.Tensor):
        # toDo could we find a better solution for proba tracking?
        #  Resampling until angle < theta? Easy on the sphere (restrain
        #  probas on the sphere pics inside a cone theta) but what do we do
        #  for other models? Ex: For the Gaussian direction getter?
        if self.normalize_directions:
            # Already normalized
            cos_angle = torch.sum(next_dirs * n_v_in, dim=1)
        else:
            norm1 = torch.linalg.norm(next_dirs, dim=-1)
            norm2 = torch.linalg.norm(n_v_in, dim=-1)
            cos_angle = torch.sum(torch.div(next_dirs, norm1[:, None]) *
                                  torch.div(n_v_in, norm2[:, None]), dim=1)

        # Resolving numerical instabilities:
        # (Converts angle to numpy)
        # Torch does not have a min() for tensor vs scalar. Using np. Ok,
        # small step.
        cos_angle = np.minimum(np.maximum(-1.0, cos_angle.cpu()), 1.0)
        angle = torch.arccos(cos_angle)

        if self.verify_opposite_direction:
            mask_angle = angle > np.pi / 2  # 90 degrees
            angle[mask_angle] = np.mod(angle[mask_angle] + np.pi, 2*np.pi)
            next_dirs[mask_angle] = - next_dirs[mask_angle]

        mask_angle = angle > self.theta
        next_dirs[mask_angle] = torch.full((3,), fill_value=torch.nan,
                                           device=self.device)

        return next_dirs

    def finalize_streamlines(self, final_lines: List[torch.Tensor],
                             v_in: List[torch.Tensor], mask):
        #  Looping. It should not be heavy.
        #  toDo. Create a tensor mask?
        for i in range(len(final_lines)):
            final_pos = final_lines[i][-1, :] + self.step_size * v_in[i]

            if (final_pos is not None and
                    mask.is_coordinate_in_bound(
                        *final_pos.cpu().numpy(),
                        space=self.space, origin=self.origin)):
                final_lines[i] = torch.vstack((final_lines[i], final_pos))

        return final_lines


class DWIMLPropagatorwithStreamlineMemory(DWIMLPropagator):
    def __init__(self, input_memory=False, **kw):
        """
        As compared to the general propagator, here, we need to send the
        whole streamline to the model in order to generate the next point's
        position. As it is the tracker's job to generally manage the memory of
        streamlines, we do not have access to these values. We need to copy
        them in memory here as long as the streamline is not finished being
        tracked.

        Parameters
        ----------
        input_memory: bool
            Remember the input value(s) at each point (in addition to the
            streamline itself, i.e. the coordinates, always saved). Warning:
            could be heavier in memory. Default: False.
        """
        super().__init__(**kw)

        self.use_input_memory = input_memory

        self.current_lines = None  # type: Union[List, None]
        # List of lines. All lines have the same number of points
        #         # as they are being propagated together.
        #         # List[list[list]]: nb_lines x (nb_points, 3).
        self.input_memory = None  # type: Union[List, None]
        # List of inputs, as formatted by the model.

    def prepare_forward(self, seeding_pos):
        self.current_lines = None
        self.input_memory = None
        return super().prepare_forward(seeding_pos)

    def prepare_backward(self, lines, forward_dir):
        # No need to invert the list of coordinates. Will be done by the
        # tracker anyway. We will update it at the next propagate() call.
        # We need to manage the input.
        if self.use_input_memory:
            # Not keeping the initial input point. Backward will start at
            # that point and will compute it again.
            self.input_memory = [torch.flip(line_input[1:, :], dims=[0])
                                 for line_input in self.input_memory]

        return super().prepare_backward(lines, forward_dir)

    def finalize_streamlines(self, final_lines: List[torch.Tensor],
                             v_in: List[torch.Tensor], mask):
        #  Looping. It should not be heavy.
        #  toDo. Create a tensor mask?
        for i in range(len(final_lines)):
            final_pos = final_lines[i][-1, :] + self.step_size * v_in[i]

            if (final_pos is not None and
                    mask.is_coordinate_in_bound(
                        *final_pos.cpu().numpy(),
                        space=self.space, origin=self.origin)):
                final_lines[i] = torch.vstack((final_lines[i], final_pos))
                if self.use_input_memory:
                    inputs = self._prepare_inputs_at_pos([final_pos])
                    self.input_memory[i] = torch.cat(
                        (self.input_memory[i], inputs[i]), dim=0)

        return final_lines

    def propagate(self, lines, v_in):
        self.current_lines = lines

        return super().propagate(lines, v_in)

    def _get_model_outputs_at_pos(self, n_pos):
        """
        Parameters
        ----------
        n_pos: list[tensor(1,3)]
            Current position coordinates for each streamline.
        """
        inputs = self._prepare_inputs_at_pos(n_pos)

        if self.use_input_memory:
            if self.input_memory is None:
                self.input_memory = inputs
            else:
                # If they all had the same lengths we could concatenate
                # everything. But during backward, they don't.
                self.input_memory = \
                    [torch.cat((self.input_memory[i], inputs[i]), dim=0)
                     for i in range(len(self.current_lines))]

        start_time = datetime.now()
        if self.use_input_memory:
            model_outputs = self._call_model_forward(self.input_memory,
                                                     self.current_lines)
        else:
            model_outputs = self._call_model_forward(inputs,
                                                     self.current_lines)
        duration_running_model = datetime.now() - start_time

        logger.debug("Time to run the model: {}"
                     .format(duration_running_model.total_seconds()))

        return model_outputs

    def _call_model_forward(self, inputs, lines):
        return self.model(inputs, lines)


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
    model: MainModelOneInput

    def __init__(self, input_volume_group: str, **kw):
        """
        Params
        ------
        input_volume_group: str
            The volume group to use as input in the model.
        """
        super().__init__(**kw)

        # Find group index in the data
        self.volume_group = self.dataset.volume_groups.index(
            input_volume_group)

        # To help prepare the inputs
        self.volume_group_str = input_volume_group

        if self.dataset.is_lazy and self.dataset.cache_size == 0:
            logger.warning("With lazy data and multiprocessing, you should "
                           "not keep cache size to zero. Data would be "
                           "loaded again at each propagation step!")

        assert self.space == Space.VOX

    def _prepare_inputs_at_pos(self, n_pos):
        """
        Prepare inputs at current position: get the volume and interpolate at
        current coordinate (possibly get the neighborhood coordinates too).

        Params
        ------
        n_pos: List[Tensor(1, 3)]
            List of n "streamlines" composed of one point.
        """
        n_pos = [pos[None, :] for pos in n_pos]
        inputs, _ = self.model.prepare_batch_one_input(
            n_pos, self.dataset, self.subj_idx, self.volume_group)

        return inputs
