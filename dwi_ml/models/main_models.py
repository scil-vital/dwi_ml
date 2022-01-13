# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil

import torch
from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs, compute_and_normalize_directions

from dwi_ml.experiment_utils.prints import TqdmLoggingHandler, \
    format_dict_to_str


class MainModelAbstract(torch.nn.Module):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """
    def __init__(self, experiment_name, normalize_directions=True,
                 neighborhood_type=None, neighborhood_radius=None):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        normalize_directions: bool
            If true, direction vectors are normalized (norm=1). If the step
            size is fixed, it shouldn't make any difference. If streamlines are
            compressed, in theory you should normalize, but you could hope that
            not normalizing could give back to the algorithm a sense of
            distance between points.
        neighborhood_type: Union[str, None]
            For usage explanation, see prepare_neighborhood_information.
        neighborhood_radius: Union[int, float, Iterable[float], None]
            For usage explanation, see prepare_neighborhood_information.
        """
        super().__init__()

        self.experiment_name = experiment_name
        self.best_model_state = None

        # Model's logging level can be changed separately from main scripts.
        self.logger = logging.getLogger('model_logger')
        self.logger.propagate = False
        self.logger.setLevel(logging.root.level)

        # Following information is actually dealt with by the data_loaders
        # (batch sampler during training and tracking_field during tracking)
        # or even by the trainers.
        # But saving the parameters directly in the model to ensure that batch
        # sampler and tracking field will both receive the same information.
        self.normalize_directions = normalize_directions

        # Possible neighbors for each input.
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_type = neighborhood_type
        self.neighborhood_points = prepare_neighborhood_vectors(
            neighborhood_type, neighborhood_radius)

        self.device = None

    def set_logger_level(self, level):
        self.logger.setLevel(level)

    def make_logger_tqdm_fitted(self):
        """Possibility to use a tqdm-compatible logger in case the model
        is used through a tqdm progress bar."""
        self.logger.addHandler(TqdmLoggingHandler())

    @property
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        return {
            'experiment_name': self.experiment_name,
            'neighborhood_type': self.neighborhood_type,
            'neighborhood_radius': self.neighborhood_radius,
            'normalize_directions': self.normalize_directions
        }

    def update_best_model(self):
        # Initialize best model
        # Uses torch's module state_dict.
        self.best_model_state = self.state_dict()

    def save(self, saving_dir):
        # Make model directory
        model_dir = os.path.join(saving_dir, "model")

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(saving_dir, "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        name = os.path.join(model_dir, "parameters.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state,
                   os.path.join(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    @classmethod
    def load(cls, loading_dir):
        """
        loading_dir: path to the trained parameters. Must contain files
            - parameters.json
            - best_model_state.pkl
        """
        # Make model directory
        model_dir = os.path.join(loading_dir)

        # Load attributes and hyperparameters from json file
        params_filename = os.path.join(model_dir, "parameters.json")
        params = json.load(open(params_filename))

        logging.debug("Loading model from saved parameters:" +
                      format_dict_to_str(params))

        best_model_filename = os.path.join(model_dir, "best_model_state.pkl")
        best_model_state = torch.load(best_model_filename)

        model = cls(**params)
        model.load_state_dict(best_model_state)  # using torch's method
        model.eval()

        return model

    def compute_loss(self, model_outputs, streamlines, device):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def format_directions(self, streamlines, device):
        targets = compute_and_normalize_directions(
            streamlines, device, self.normalize_directions)
        return targets

    def get_tracking_direction_det(self, model_outputs):
        """
        This needs to be implemented in order to use the model for
        generative tracking, as in dwi_ml.tracking.tracker_abstract.

        Probably calls a directionGetter.get_tracking_directions_det.

        Returns
        -------
        next_dir: array(3,)
            Numpy array with x,y,z value.
        """
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs):
        """
        This needs to be implemented in order to use the model for
        generative tracking, as in dwi_ml.tracking.tracker_abstract.

        Probably calls a directionGetter.sample_tracking_directions_prob.

        Returns
        -------
        next_dir: array(3,)
            Numpy array with x,y,z value.
        """
        raise NotImplementedError


class MainModelWithPD(MainModelAbstract):
    def __init__(self, experiment_name, nb_previous_dirs,
                 normalize_directions=False,
                 neighborhood_type=None, neighborhood_radius=None):
        """
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received.
        """
        super().__init__(experiment_name, normalize_directions,
                         neighborhood_type, neighborhood_radius)
        self.nb_previous_dirs = nb_previous_dirs

    @property
    def params(self):
        p = super().params
        p.update({
            'nb_previous_dirs': self.nb_previous_dirs,
        })
        return p

    def compute_loss(self, outputs, targets, device):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def get_tracking_direction_det(self, model_outputs):
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs):
        raise NotImplementedError

    def format_previous_dirs(self, all_streamline_dirs, device,
                             point_idx=None):
        """
        Formats the previous dirs. See compute_n_previous_dirs for a
        description of parameters.
        """
        if self.nb_previous_dirs == 0:
            return None

        n_previous_dirs = compute_n_previous_dirs(
            all_streamline_dirs, self.nb_previous_dirs, device=device,
            point_idx=point_idx)

        return n_previous_dirs
