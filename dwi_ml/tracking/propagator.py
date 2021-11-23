# -*- coding: utf-8 -*-
import logging
import math

import numpy as np
import torch
import torch.nn.functional as f

from scilpy.tracking.propagator import AbstractPropagator

from dwi_ml.tracking.tracking_field import DWIMLAbstractTrackingField, \
                                           DWIMLTrackingFieldOneInputAndPD


class DWIMLAbstractPropagator(AbstractPropagator):
    """
    Abstract class for propagator object. Responsible for sampling the final
    direction from the model outputs offered by the tracking field, and
    for the propagation through Runge-Kutta integration.

    To allow easier parent/child class management, the algo choice (det or
    prob) is added as a parameter instead of creating a deterministic child and
    a probabilistic child.

    Theta is included here rather than in the tracking field like in scilpy.
    """
    def __init__(self, tracking_field: DWIMLAbstractTrackingField,
                 step_size: float, rk_order: int, algo: str, theta: float):
        """
        Parameters
        ----------
        tracking_field: dwi_ml tracking field object
            TrackingField object on which the tracking is done. (Contains the
            data and functions to get the next direction inside an angle
            theta).
        step_size: float
            The step size for tracking.
        rk_order: int
            Order for the Runge Kutta integration.
        algo: str
            'det' or 'prob'
        theta: float
            Maximum angle (radians) between two steps.
        """
        # Send a warning because our tracking field is not a child of
        # scilpy's AbstractTracking field but ok, will work!
        super().__init__(tracking_field, step_size, rk_order)

        self.algo = algo
        if algo not in ['det', 'prob']:
            raise ValueError("Propagator's algo should be 'det' or 'prob'.")

        self.theta = theta

    def _sample_init_direction(self, pos):
        first_dir = self._sample_next_direction(pos, None)
        return first_dir, -first_dir

    def _sample_next_direction(self, pos, v_in):
        """
        Run the model to get the direction. Tracking field is either
        from a regression or classification model, but in both cases, outputs
        is a direction, or None if the direction is more than theta angle
        from v_in. (Then, direction will be considered invalid in
        _sample_next_direction_or_go_straight and v_in will be followed
        instead, for a maximum or X invalid steps.)

        Parameters
        ----------
        pos: ndarray (3,) or List[ndarray (3,)]
            Initial tracking position or batch of positions.
        v_in: ndarray (3,) or List[ndarray (3,)]
            Previous direction or batch of previous directions.

        Return
        ------
        next_dir : ndarray (3,) or List[ndarray (3,)]
            The next direction or batch of next directions.
        """
        logging.debug("----------> Running the model at current position.")
        # See the propagator. Last v_in has not been added to the previous_dirs
        # yet; during runge-kutta evaluation, temporary directions are computed
        model_outputs = self._get_model_outputs(pos)

        logging.debug("----------> Sampling a direction from the model's "
                      "output through the tracking field. Model: {}"
                      .format(type(self.tracking_field.model)))
        if self.algo == 'prob':
            next_dir = \
                self.tracking_field.model.sample_tracking_direction_prob(
                    model_outputs)
        else:
            next_dir = \
                self.tracking_field.model.get_tracking_direction_det(
                    model_outputs)

        logging.debug("----------> Next dir = {}."
                      .format(next_dir))
        if v_in is not None and len(v_in) > 0:
            next_dir = self._verify_angle(next_dir, v_in)

        # Need to detach grad to be able to convert to numpy array
        return next_dir

    def _get_model_outputs(self, pos):
        """
        Re-implement in a child class if your model needs more information
        than just the current position.
        """
        return self.tracking_field.get_model_outputs_at_pos(pos)

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
        angle = np.arccos(cos_angle)
        logging.debug("----------> Curvature = {}".format(math.degrees(angle)))
        if angle > self.theta:
            return None
        return next_dir

    def is_voxmm_in_bound(self, pos, origin):
        """
        Compared to scilpy, needs to be re-implemented. Data volume is loaded
        from lazy or non-lazy data.

        Depends on the choice of input. Here, taking input #0, input volumes
        probably have the same bounds. Reimplement in a child class if that is
        not the case.
        """
        mri_data = self.tracking_field.dataset.mri_data_list[0]
        return mri_data.as_data_volume.is_voxmm_in_bound(*pos, origin)


class DWIMLPropagatorOneInputAndPD(DWIMLAbstractPropagator):
    """
    Associated to a model with inputs:
        - inputs: a volume from the hdf5 + (neighborhood)
        - previous_dirs: the n previous dirs.
    """
    def __init__(self, tracking_field: DWIMLTrackingFieldOneInputAndPD,
                 step_size: float, rk_order: int, algo: str, theta: float):
        super().__init__(tracking_field, step_size, rk_order, algo,
                         theta)

        # The list of previous dirs will be known by the tracker but to keep
        # memory of this in the tracker we would have to reimplement a lot of
        # scilpy's big methods such as propagator.propagate and
        # tracker._propagate_line. Easier to keep memory of it here.
        # But in tracker._get_line_both_directions, need to tell this class
        # to reverse the previous dirs when starting backward session.
        self.previous_dirs = None

    def start_backward(self):
        """
        Will need to be called between forward and backward tracking.
        """
        self.previous_dirs = self.previous_dirs.reverse()

    def initialize(self, pos, track_forward_only):
        # Empty previous dirs when starting a new streamline
        self.previous_dirs = []

        return super().initialize(pos, track_forward_only)

    def _get_model_outputs(self, pos):
        # Compared to super, adding previous dirs.
        return self.tracking_field.get_model_outputs_at_pos(
            pos, self.previous_dirs)

    def propagate(self, pos, v_in):
        logging.debug("-------> New step. Will check various possible "
                      "next directions based on Runge-Kutta integration.")
        logging.debug("         Current previous_dirs: {}"
                      .format(self.previous_dirs))
        # Summary:
        # super()'s propagate = scilpy's. does the Runge-kutta integration.
        # For each direction measure, it uses _sample_next_direction,
        # reimplemented in dwi_ml's abstract class, which runs the model
        # (calling sub-method _get_model_outputs, see current class to see
        # how previous_dirs are added.).
        new_pos, new_dir, is_direction_valid = super().propagate(pos, v_in)
        self.previous_dirs.append(new_dir)
        logging.debug("-------> Final choice: New pos = {}. "
                      "Len(previous_dirs) = {}"
                      .format(new_pos, len(self.previous_dirs)))

        return new_pos, new_dir, is_direction_valid

    def is_voxmm_in_bound(self, pos, origin):
        # Compared to scilpy, needs to be re-implemented. Data volume is
        # loaded from lazy or non-lazy data
        # Using the volume of volume_group associated to the model's inputs.
        data_volume = self.tracking_field.dataset.mri_data_list[
            self.tracking_field.volume_group].as_data_volume
        return data_volume.is_voxmm_in_bound(*pos, origin)
