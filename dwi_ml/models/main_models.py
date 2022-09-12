# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
from datetime import datetime

import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence, \
    unpack_sequence

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs, compute_and_normalize_directions
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
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

        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)

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
            self.neighborhood_points is not None else 0

        self.device = None

    def move_to(self, device):
        """
        Careful. Calling model.to(a_device) does not influence the self.device.
        Prefer this method for easier management.
        """
        self.to(device)
        self.device = device

    @property
    def params_for_checkpoint(self):
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

    @property
    def params_for_json_prints(self):
        return self.params_for_checkpoint

    def save_params_and_state(self, model_dir):
        model_state = self.state_dict()

        # If a model was already saved, back it up and erase it after saving
        # the new.
        to_remove = None
        if os.path.exists(model_dir):
            to_remove = os.path.join(model_dir, "..", "model_old")
            shutil.move(model_dir, to_remove)
        os.makedirs(model_dir)

        # Save attributes
        name = os.path.join(model_dir, "parameters.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params_for_checkpoint, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(model_state, os.path.join(model_dir, "model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    @classmethod
    def load_params_and_state(cls, model_dir, log_level=logging.WARNING):
        """
        Params
        -----
        loading_dir: path
            Path to the trained parameters, either from the latest checkpoint
            or from the best model folder. Must contain files
            - parameters.json
            - model_state.pkl
        """
        # Load attributes and hyperparameters from json file
        params_filename = os.path.join(model_dir, "parameters.json")
        params = json.load(open(params_filename))

        logger.setLevel(log_level)
        logger.debug("Loading model from saved parameters:" +
                     format_dict_to_str(params))

        model_state_file = os.path.join(model_dir, "model_state.pkl")
        model_state = torch.load(model_state_file)

        model = cls(log_level=log_level, **params)
        model.load_state_dict(model_state)  # using torch's method

        # By default, setting to eval state. If this will be used by the
        # trainer, it will call model.train().
        model.eval()

        return model

    def compute_loss(self, model_outputs, streamlines):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def format_directions(self, streamlines):
        """
        Simply calls compute_and_normalize_directions, but with model's device
        and options.
        """
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
        next_dir: list[array(3,)]
            Numpy arrays with x,y,z value, one per streamline data point.
        """
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs):
        """
        This needs to be implemented in order to use the model for
        generative tracking, as in dwi_ml.tracking.tracker_abstract.

        Probably calls a directionGetter.sample_tracking_directions_prob.

        Returns
        -------
        next_dir: list[array(3,)]
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
        self.prev_dirs_embedding_size = prev_dirs_embedding_size

        if self.nb_previous_dirs > 0:
            if self.prev_dirs_embedding_key not in keys_to_embeddings.keys():
                raise ValueError("Embedding choice for previous dirs not "
                                 "understood: {}"
                                 .format(self.prev_dirs_embedding_key))

            if prev_dirs_embedding_size is None:
                self.prev_dirs_embedding_size = nb_previous_dirs * 3

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
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'nb_previous_dirs': self.nb_previous_dirs,
            'prev_dirs_embedding_key': self.prev_dirs_embedding_key,
            'prev_dirs_embedding_size': self.prev_dirs_embedding_size,
        })
        return p

    def compute_loss(self, outputs, targets):
        # Probably something like:
        # targets = self._format_directions(streamlines)
        # Then compute loss based on model.
        raise NotImplementedError

    def compute_and_embed_previous_dirs(self, dirs,
                                        unpack_results: bool = True,
                                        point_idx=None):
        """
        Runs the self.prev_dirs_embedding layer, if instantiated, and returns
        the model's output. Else, returns the data as is.

        Params
        ------
        dirs: Union[List, torch.tensor],
            Batch all streamline directions. If it is a tensor, it should be of
            size [nb_points, 3].
            If it is a list, length of the list is the number of streamlines in
            the batch. Each tensor is as described above. The batch will be
            packed and embedding will be ran on resulting tensor.
        unpack_results: bool
            If data was a list, unpack the model's outputs before returning.
            Default: True. Hint: skipping unpacking can be useful if you want
            to concatenate this embedding to your input's packed sequence's
            embedding.
        point_idx: int
            Point of the streamline for which to compute the previous dirs.

        Returns
        -------
        n_prev_dirs_embedded: Union[PackedSequence, Tensor]
            The previous dirs.
        """
        if self.nb_previous_dirs == 0:
            return None
        else:
            # Formatting the n previous dirs for all points.
            n_prev_dirs = self.format_previous_dirs(dirs, point_idx=point_idx)

            if not point_idx:
                # Not keeping the last point: only useful to get the last
                # direction (ex, last target), but won't be used as an input.
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

                    n_prev_dirs_embedded = unpack_sequence(
                        n_prev_dirs_embedded_packed)

                return n_prev_dirs_embedded

    def get_tracking_direction_det(self, model_outputs,
                                   streamline_lengths=None):
        raise NotImplementedError

    def sample_tracking_direction_prob(self, model_outputs,
                                       streamline_lengths=None):
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


class MainModelOneInput(MainModelAbstract):
    def __init__(self, **kw):
        super().__init__(**kw)

    @staticmethod
    def prepare_batch_one_input(
            streamlines, subset, subj, input_group_idx,
            neighborhood_points, device, prepare_mask=False):
        """
        Params
        ------
        streamlines: list, The streamlines, IN VOXEL SPACE, CORNER ORIGIN
        subjset: MultisubjectSubset, The dataset
        subj: str, The subject id
        input_groupd_idx: int, The volume group
        neighborhood_points: list, The neighborhood coordinates
        device: Torch device
        prepare_mask: bool, If true, return a mask of chosen coordinates
        logger: logging logger.
        """
        start_time = datetime.now()

        # Flatten = concatenate signal for all streamlines to process
        # faster.
        flat_subj_x_coords = np.concatenate(streamlines, axis=0)

        # Getting the subject's volume and sending to CPU/GPU
        # If data is lazy, get volume from cache or send to cache if
        # it wasn't there yet.
        data_tensor = subset.get_volume_verify_cache(
            subj, input_group_idx, device=device, non_blocking=True)

        # Prepare the volume data, possibly adding neighborhood
        # (Thus new coords_torch possibly contain the neighborhood points)
        # Coord_clipped contain the coords after interpolation
        subj_x_data, coords_torch = interpolate_volume_in_neighborhood(
            data_tensor, flat_subj_x_coords, neighborhood_points,
            device)

        # Split the flattened signal back to streamlines
        lengths = [len(s) for s in streamlines]
        subj_x_data = subj_x_data.split(lengths)
        duration_inputs = datetime.now() - start_time
        logger.debug("Time to prepare the inputs ({} streamlines, total {} "
                     "points): {} s".format(len(streamlines), sum(lengths),
                                            duration_inputs.total_seconds()))

        input_mask = None
        if prepare_mask:
            print("DEBUGGING MODE. Returning batch_streamlines "
                  "and mask together with inputs.")

            # Clipping used coords (i.e. possibly with neighborhood)
            # outside volume
            lower = torch.as_tensor([0, 0, 0], device=device)
            upper = torch.as_tensor(data_tensor.shape[:3], device=device)
            upper -= 1
            coords_to_idx_clipped = torch.min(
                torch.max(torch.floor(coords_torch).long(), lower),
                upper)
            input_mask = torch.tensor(np.zeros(data_tensor.shape[0:3]))
            for s in range(len(coords_torch)):
                input_mask.data[tuple(coords_to_idx_clipped[s, :])] = 1

        return subj_x_data, input_mask
