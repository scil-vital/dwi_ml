# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil

import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence, \
    unpack_sequence

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs, compute_and_normalize_directions
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings

logger = logging.getLogger('model_logger')


class MainModelAbstract(torch.nn.Module):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """
    def __init__(self, experiment_name: str, normalize_directions: bool = True,
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level):
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
            distance between points. Default: True.
        neighborhood_type: str
            For usage explanation, see prepare_neighborhood_information.
            Default: None.
        neighborhood_radius: Union[int, float, Iterable[float]]
            For usage explanation, see prepare_neighborhood_information.
            Default: None.
        """
        super().__init__()

        self.experiment_name = experiment_name
        self.best_model_state = None

        # Trainer's logging level can be changed separately from main
        # scripts.
        self.logger = logger
        self.logger.setLevel(log_level)

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
        self.nb_neighbors = len(self.neighborhood_points) if \
            self.neighborhood_points else 0

        self.device = None

    @staticmethod
    def set_logger_state(level):
        logger.setLevel(level)

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
        model.update_best_model()
        model.eval()

        return model

    def compute_loss(self, model_outputs, streamlines, device):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def format_directions(self, streamlines):
        targets = compute_and_normalize_directions(
            streamlines, self.device, self.normalize_directions)
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
    """
    Adds tools to work with previous directions. Prepares a layer for previous
    directions embedding, and a tool method for direction formatting.

    Hint: In your forward method, first concantenate your input with the result
    of the previous directions embedding layer!
    """
    def __init__(self, experiment_name: str, nb_previous_dirs: int = 0,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = None,
                 normalize_directions: bool = True,
                 neighborhood_type: str = None, neighborhood_radius=None,
                 log_level=logging.root.level):
        """
        nb_previous_dirs: int
            Number of previous direction (i.e. [x,y,z] information) to be
            received. Default: 0.
        prev_dirs_embedding_size: int
            Dimension of the final vector representing the previous directions
            (no matter the number of previous directions used).
            Default: nb_previous_dirs * 3.
        prev_dirs_embedding_key: str,
            Key to an embedding class (one of
            dwi_ml.models.embeddings_on_tensors.keys_to_embeddings).
            Default: None (no previous directions added).
        """
        super().__init__(experiment_name, normalize_directions,
                         neighborhood_type, neighborhood_radius,
                         log_level)

        self.nb_previous_dirs = nb_previous_dirs
        self.prev_dirs_embedding_key = prev_dirs_embedding_key

        if self.nb_previous_dirs > 0:
            if self.prev_dirs_embedding_key not in keys_to_embeddings.keys():
                raise ValueError("Embedding choice for previous dirs not "
                                 "understood: {}"
                                 .format(self.prev_dirs_embedding_key))

            self.prev_dirs_embedding_size = \
                prev_dirs_embedding_size if not None else nb_previous_dirs * 3
            prev_dirs_emb_cls = keys_to_embeddings[prev_dirs_embedding_key]
            # Preparing layer!
            self.prev_dirs_embedding = prev_dirs_emb_cls(
                input_size=nb_previous_dirs * 3,
                output_size=self.prev_dirs_embedding_size)
        else:
            self.prev_dirs_embedding_size = None
            if prev_dirs_embedding_size:
                logging.warning("Previous dirs embedding size was defined but "
                                "no previous directions are used!")
            self.prev_dirs_embedding = None

    @property
    def params(self):
        p = super().params
        p.update({
            'nb_previous_dirs': self.nb_previous_dirs,
            'prev_dirs_embedding_key': self.prev_dirs_embedding_key,
            'prev_dirs_embedding_size': self.prev_dirs_embedding_size,
        })
        return p

    def compute_loss(self, outputs, targets, device):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def run_prev_dirs_embedding_layer(self, dirs,
                                      unpack_results: bool = True):
        """
        Runs the self.prev_dirs_embedding layer, if instantiated, and returns
        the model's output. Else, returns the data as is.

        Params
        ------
        n_prev_dirs: Union[List, torch.tensor],
            Batch of n past directions. If it is a tensor, it should be of
            size [nb_points, nb_previous_dirs * 3].
            If it is a list, length of the list is the number of streamlines in
            the batch. Each tensor is as described above. The batch will be
            packed and embedding will be ran on resulting tensor.
        device: torch device
        unpack_results: bool
            If data was a list, unpack the model's outputs before returning.
            Default: True. Hint: skipping unpacking can be useful if you want
            to concatenate this embedding to your input's packed sequence's
            embedding.
        """
        if self.nb_previous_dirs == 0:
            return None
        else:
            # Formatting the n previous dirs for all points.
            n_prev_dirs = self.format_previous_dirs(dirs, self.device)

            # Not keeping the last point: only useful to get the last direction
            # (ex, last target), but won't be used as an input.
            n_prev_dirs = [s[:-1] for s in n_prev_dirs]

            if self.prev_dirs_embedding is None:
                return n_prev_dirs
            else:
                is_list = isinstance(n_prev_dirs, list)
                if is_list:
                    # Using Packed_sequence's tensor.
                    n_prev_dirs_packed = pack_sequence(n_prev_dirs,
                                                       enforce_sorted=False)

                    n_prev_dirs = n_prev_dirs_packed.data
                    n_prev_dirs.to(self.device)

                n_prev_dirs_embedded = self.prev_dirs_embedding(n_prev_dirs)

                if is_list and unpack_results:
                    # Packing back to unpack correctly
                    batch_sizes = n_prev_dirs_packed.batch_sizes
                    sorted_indices = n_prev_dirs_packed.sorted_indices
                    unsorted_indices = n_prev_dirs_packed.unsorted_indices
                    n_prev_dirs_embedded_packed = \
                        PackedSequence(n_prev_dirs_embedded, batch_sizes,
                                       sorted_indices, unsorted_indices)

                    n_prev_dirs_embedded, _ = unpack_sequence(
                        n_prev_dirs_embedded_packed)

                return n_prev_dirs_embedded

    def get_tracking_direction_det(self, model_outputs):
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs):
        raise NotImplementedError

    def format_previous_dirs(self, all_streamline_dirs, point_idx=None):
        """
        Formats the previous dirs. See compute_n_previous_dirs for a
        description of parameters.
        """
        if self.nb_previous_dirs == 0:
            return None

        n_previous_dirs = compute_n_previous_dirs(
            all_streamline_dirs, self.nb_previous_dirs,
            point_idx=point_idx, device=self.device)

        return n_previous_dirs
