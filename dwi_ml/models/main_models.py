# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import shutil
from typing import List, Union

import numpy as np
import torch
from torch import Tensor

from dwi_ml.data.dataset.multi_subject_containers import MultisubjectSubset
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.data.processing.space.neighborhood import prepare_neighborhood_vectors
from dwi_ml.experiment_utils.prints import format_dict_to_str
from dwi_ml.io_utils import add_resample_or_compress_arg
from dwi_ml.models.direction_getter_models import keys_to_direction_getters, \
    AbstractDirectionGetterModel
from dwi_ml.models.embeddings_on_tensors import keys_to_embeddings
from dwi_ml.models.utils.direction_getters import add_direction_getter_args

logger = logging.getLogger('model_logger')


class MainModelAbstract(torch.nn.Module):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """
    def __init__(self, experiment_name: str, step_size: float = None,
                 compress_lines: float = False, log_level=logging.root.level):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        step_size : float
            Constant step size that every streamline should have between points
            (in mm). Default: None.
            Note that you probably already fixed a step size when
            creating your dataset, but you could use a different one here if
            you wish. Training / testing / tracking will be performed on
            resampled streamlined. When using an existing model in various
            scripts, you will often have the option to modify this value, but
            it is probably not recommanded.
        compress_streamlines: float
            If set, compress streamlines to that tolerance error. Cannot be
            used together with step_size. Once again, the choice can be
            different here than chosen when creating the hdf5. Default: False.
            * Do not counfound with direction getters' compress_loss param.
        log_level: str
            Level of the model logger. Default: root's level.
        """
        super().__init__()

        self.experiment_name = experiment_name

        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)

        self.device = None

        # To tell our trainer what to send to the forward / loss methods.
        self.forward_uses_streamlines = False
        self.loss_uses_streamlines = False
        
        # To tell our batch loader how to resample streamlines during training
        # (should also be the step size during tractography).
        if step_size and compress_lines:
            raise ValueError("You may choose either resampling or compressing,"
                             "but not both.")
        elif step_size and step_size <= 0:
            raise ValueError("Step size can't be 0 or less!")
            # Note. When using
            # scilpy.tracking.tools.resample_streamlines_step_size, a warning
            # is shown if step_size < 0.1 or > np.max(sft.voxel_sizes), saying
            # that the value is suspicious. Not raising the same warnings here
            # as you may be wanting to test weird things to understand better
            # your model.
        self.step_size = step_size
        self.compress_lines = compress_lines

        # Adding a context. Most models in here act differently
        # during training (ex: no loss at the last coordinate = we skip it)
        # vs during tracking (only the last coordinate is important) vs during
        # visualisation (the whole streamline is important).
        self._context = None

    @staticmethod
    def add_args_main_model(p):
        add_resample_or_compress_arg(p)

    def set_context(self, context):
        assert context in ['training', 'tracking']
        self._context = context

    @property
    def context(self):
        return self._context

    def move_to(self, device):
        """
        Careful. Calling model.to(a_device) does not influence the self.device.
        Prefer this method for easier management.
        """
        self.to(device, non_blocking=True)
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
            'step_size': self.step_size,
            'compress_lines': self.compress_lines,
        }

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
        with open(params_filename, 'r') as json_file:
            params = json.load(json_file)

        logger.setLevel(log_level)
        logger.debug("Loading model from saved parameters:" +
                     format_dict_to_str(params))

        model_state_file = os.path.join(model_dir, "model_state.pkl")
        model_state = torch.load(model_state_file)

        params.update(log_level=log_level)
        model = cls(**params)
        model.load_state_dict(model_state)  # using torch's method

        # By default, setting to eval state. If this will be used by the
        # trainer, it will call model.train().
        model.eval()

        return model

    def compute_loss(self, *model_outputs, **kw):
        raise NotImplementedError

    def forward(self, *inputs, **kw):
        raise NotImplementedError


class ModelWithNeighborhood(MainModelAbstract):
    """
    Adds tools to work with neighborhoods.
    """
    def __init__(self, neighborhood_type: str = None,
                 neighborhood_radius=None, **kw):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        log_level: str
            Level of the model logger.
        neighborhood_type: str
            For usage explanation, see prepare_neighborhood_information.
            Default: None.
        neighborhood_radius: Union[int, float, Iterable[float]]
            For usage explanation, see prepare_neighborhood_information.
            Default: None.
        """
        super().__init__(**kw)

        # Possible neighbors for each input.
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_type = neighborhood_type
        self.neighborhood_vectors = prepare_neighborhood_vectors(
            neighborhood_type, neighborhood_radius)

        # Reminder. nb neighbors does not include origin.
        self.nb_neighbors = len(self.neighborhood_vectors) if \
            self.neighborhood_vectors is not None else 0

    def move_to(self, device):
        super().move_to(device)
        if self.neighborhood_vectors is not None:
            self.neighborhood_vectors = self.neighborhood_vectors.to(
                device, non_blocking=True)

    @staticmethod
    def add_neighborhood_args_to_parser(p: argparse.PARSER):
        # The experiment name and log level are dealt with by the
        # main experiment parameters.
        p.add_argument(
            '--neighborhood_type', choices=['axes', 'grid'],
            help="If set, add neighborhood vectors either with the 'axes' "
                 "or 'grid' option to \nthe input(s).\n"
                 "- 'axes': lies on a sphere. Uses a list of 6 positions (up, "
                 "down, left, right, \nbehind, in front) at exactly "
                 "neighborhood_radius voxels from tracking point.\n"
                 "- 'grid': Uses a list of vectors pointing to points "
                 "surrounding the origin \nthat mimic the original voxel "
                 "grid, in voxel space.")
        p.add_argument(
            '--neighborhood_radius', type=float,
            metavar='r', nargs='+',
            help="- With type 'axes', radius must be a float or a list[float] "
                 "(it will then be a \nmulti-radius neighborhood (lying on "
                 "concentring spheres).\n"
                 "- With type 'grid': radius must be a single int value, the "
                 "radius in number of \nvoxels. Ex: with radius 1, this is "
                 "26 points. With radius 2, it's 124 points.")

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'neighborhood_type': self.neighborhood_type,
            'neighborhood_radius': self.neighborhood_radius,
        })
        return p


class ModelWithPreviousDirections(MainModelAbstract):
    """
    Adds tools to work with previous directions. Prepares a layer for previous
    directions embedding, and a tool method for direction formatting.
    """
    def __init__(self, nb_previous_dirs: int = 0,
                 prev_dirs_embedding_size: int = None,
                 prev_dirs_embedding_key: str = None,
                 normalize_prev_dirs: bool = True, **kw):
        """
        Params
        ------
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
        normalize_prev_dirs: bool
            If true, direction vectors are normalized (norm=1) when computing
            the previous direction.
        """
        super().__init__(**kw)

        self.nb_previous_dirs = nb_previous_dirs
        self.prev_dirs_embedding_key = prev_dirs_embedding_key
        self.prev_dirs_embedding_size = prev_dirs_embedding_size
        self.normalize_prev_dirs = normalize_prev_dirs

        if self.nb_previous_dirs > 0:
            if (prev_dirs_embedding_key is not None and
                    prev_dirs_embedding_size is None):
                raise ValueError(
                    "To use an embedding class, you must provide its output size")
            if self.prev_dirs_embedding_key not in keys_to_embeddings.keys():
                raise ValueError("Embedding choice for previous dirs not "
                                 "understood: {}. It should be one of {}"
                                 .format(self.prev_dirs_embedding_key,
                                         keys_to_embeddings.keys()))

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

        # To tell our trainer what to send to the forward / loss methods.
        if nb_previous_dirs > 0:
            self.forward_uses_streamlines = True

    @staticmethod
    def add_args_model_with_pd(p):
        p.add_argument(
            '--nb_previous_dirs', type=int, default=0, metavar='n',
            help="Concatenate the n previous streamline directions to the "
                 "input vector. \nDefault: 0")
        p.add_argument(
            '--prev_dirs_embedding_key', choices=keys_to_embeddings.keys(),
            default='no_embedding',
            help="Type of model for the previous directions embedding layer.\n"
                 "Default: no_embedding (identity model).")
        p.add_argument(
            '--prev_dirs_embedding_size', type=int, metavar='s',
            help="Size of the output after passing the previous dirs through "
                 "the embedding \nlayer. (Total size. Ex: "
                 "--nb_previous_dirs 3, --prev_dirs_embedding_size 8 \n"
                 "would compact 9 features into 8.) "
                 "Default: nb_previous_dirs*3.")
        p.add_argument(
            '--normalize_prev_dirs', action='store_true',
            help="If true, normalize the previous directions (before the "
                 "embedding layer,\n if any, and before adding to the input.")

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'nb_previous_dirs': self.nb_previous_dirs,
            'prev_dirs_embedding_key': self.prev_dirs_embedding_key,
            'prev_dirs_embedding_size': self.prev_dirs_embedding_size,
            'normalize_prev_dirs': self.normalize_prev_dirs
        })
        return p

    def forward(self, inputs, target_streamlines: List[torch.tensor], **kw):
        """
        Params
        ------
        inputs: Any
            Batch of inputs. Shape: [nb_points, nb_features].
        target_streamlines: List[torch.tensor]
            Directions computed from the streamlines. Not normalized yet.
        """
        # Hints : Should start with:

        # target_dirs = compute_directions(target_streamlines)
        # n_prev_dirs_embedded = self.normalize_and_embed_previous_dirs(
        #       target_dirs)

        # Concatenate with inputs
        # Ex: inputs = torch.cat((inputs, n_prev_dirs_embedded), dim=-1)
        raise NotImplementedError


class MainModelOneInput(MainModelAbstract):
    def prepare_batch_one_input(self, streamlines, subset: MultisubjectSubset,
                                subj, input_group_idx, prepare_mask=False):
        """
        These params are passed by either the batch loader or the propagator,
        which manage the data.

        Params
        ------
        streamlines: list[Tensor]
            The streamlines, IN VOXEL SPACE, CORNER ORIGIN.
            Tensors are of shape (nb points, 3).
        subset: MultisubjectSubset
            The dataset.
        subj: str
            The subject id.
        input_groupd_idx: int
            The volume group.
        prepare_mask: bool
            If True, return a mask of chosen coordinates (DEBUGGING MODE).

        Returns
        -------
        input_data: List[Tensor]
            One input tensor per streamline. Each input is of shape
            [nb_point x nb_features].
        """
        # Flatten = concatenate signal for all streamlines to process
        # faster.
        flat_subj_x_coords = torch.cat(streamlines, dim=0)

        # Getting the subject's volume (creating it directly on right device)
        # If data is lazy, get volume from cache or send to cache if
        # it wasn't there yet.
        data_tensor = subset.get_volume_verify_cache(
            subj, input_group_idx, device=self.device)

        # Prepare the volume data
        # Coord_torch contain the coords after interpolation, possibly clipped
        # to volume bounds.
        if isinstance(self, ModelWithNeighborhood):
            # Adding neighborhood.
            subj_x_data, coords_torch = interpolate_volume_in_neighborhood(
                data_tensor, flat_subj_x_coords, self.neighborhood_vectors)
        else:
            subj_x_data, coords_torch = interpolate_volume_in_neighborhood(
                data_tensor, flat_subj_x_coords, None)

        # Split the flattened signal back to streamlines
        lengths = [len(s) for s in streamlines]
        subj_x_data = list(subj_x_data.split(lengths))

        if prepare_mask:
            logging.warning("Model OneInput: DEBUGGING MODE. Returning "
                            "batch_streamlines and mask together with inputs.")

            # Clipping used coords (i.e. possibly with neighborhood)
            # outside volume
            lower = torch.as_tensor([0, 0, 0], device=self.device)
            upper = torch.as_tensor(data_tensor.shape[:3], device=self.device)
            upper -= 1
            coords_to_idx_clipped = torch.min(
                torch.max(torch.floor(coords_torch).long(), lower), upper)
            input_mask = torch.as_tensor(np.zeros(data_tensor.shape[0:3]))
            for s in range(len(coords_torch)):
                input_mask.data[tuple(coords_to_idx_clipped[s, :])] = 1

            return subj_x_data, input_mask
        return subj_x_data


class ModelWithDirectionGetter(MainModelAbstract):
    """
    Adding typical options for models intended for learning to track.
    - Last layer should be a direction getter.
    - Added option to save the estimated outputs after a batch to compare
      with targets (in the case of regression).
    """
    def __init__(self, dg_key: str = 'cosine-regression',
                 dg_args: dict = None, **kw):
        """
        Params
        ------
        dg_key: str
            Key to a direction getter class (one of
            dwi_ml.direction_getter_models.keys_to_direction_getters).
            Default: Default: 'cosine-regression'.
        dg_args: dict
            Arguments necessary for the instantiation of the chosen direction
            getter.
        """
        super().__init__(**kw)

        # Instantiating direction getter
        # Preparing value here but not instantiating;
        # typically, user will need to know his model output size to call
        # this with correct input size. Waiting.
        self.dg_key = dg_key
        self.dg_args = dg_args or {}
        self.direction_getter = None  # type: AbstractDirectionGetterModel
        if self.dg_key not in keys_to_direction_getters.keys():
            raise ValueError("Direction getter choice not understood: {}"
                             .format(self.positional_encoding_key))

        # To tell our trainer what to send to the forward / loss methods.
        self.loss_uses_streamlines = True

    def set_context(self, context):
        assert context in ['training', 'tracking', 'visu']
        self._context = context

    def instantiate_direction_getter(self, dg_input_size):
        direction_getter_cls = keys_to_direction_getters[self.dg_key]
        self.direction_getter = direction_getter_cls(
            input_size=dg_input_size, **self.dg_args)

    @staticmethod
    def add_args_tracking_model(p):
        add_direction_getter_args(p)

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'dg_key': self.dg_key,
            'dg_args': self.dg_args,
        })
        return p

    def get_tracking_directions(self, model_outputs: Tensor, algo: str,
                                eos_stopping_thresh: Union[float, str]):
        """
        Params
        ------
        model_outputs: Tensor
            Our model's previous layer's output.
        algo: str
            'det' or 'prob'.
        eos_stopping_thresh: float or 'max'

        Returns
        -------
        next_dir: torch.Tensor
            A tensor of shape [n, 3] with the next direction for each output.
        """
        dirs = self.direction_getter.get_tracking_directions(
            model_outputs, algo, eos_stopping_thresh)
        return dirs

    def compute_loss(self, model_outputs: List[Tensor], target_streamlines,
                     average_results=True, **kw):
        return self.direction_getter.compute_loss(
            model_outputs, target_streamlines, average_results)

    def move_to(self, device):
        super().move_to(device)
        self.direction_getter.move_to(device)
