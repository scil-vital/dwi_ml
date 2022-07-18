# -*- coding: utf-8 -*-
import argparse
import json
import logging
import os
import shutil
from datetime import datetime
from typing import List, Union

from dipy.io.stateful_tractogram import StatefulTractogram
from dipy.io.streamline import save_tractogram
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence, PackedSequence, \
    unpack_sequence

from dwi_ml.data.processing.space.neighborhood import \
    prepare_neighborhood_vectors
from dwi_ml.data.processing.streamlines.post_processing import \
    compute_n_previous_dirs, normalize_directions
from dwi_ml.data.processing.volume.interpolation import \
    interpolate_volume_in_neighborhood
from dwi_ml.experiment_utils.prints import format_dict_to_str
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
    def __init__(self, experiment_name: str,
                 log_level=logging.root.level):
        """
        Params
        ------
        experiment_name: str
            Name of the experiment
        log_level: str
            Level of the model logger.
        """
        super().__init__()

        self.experiment_name = experiment_name

        # Trainer's logging level can be changed separately from main
        # scripts.
        logger.setLevel(log_level)

        self.device = None

        # To tell our trainer what to send to the forward / loss methods.
        self.model_uses_streamlines = False
        self.model_uses_dirs = False

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
        }

    @property
    def params_for_json_prints(self):
        """
        Adding more information, if necessary, for loggings.
        """
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

        params.update(log_level=log_level)
        model = cls(**params)
        model.load_state_dict(model_state)  # using torch's method

        # By default, setting to eval state. If this will be used by the
        # trainer, it will call model.train().
        model.eval()

        return model

    def compute_loss(self, *model_outputs, **kw):
        raise NotImplementedError

    def format_directions(self, streamlines):
        """
        Simply calls compute_and_normalize_directions, but with model's device
        and options.
        """
        targets = compute_and_normalize_directions(
            streamlines, self.device, self.normalize_directions)
        return targets

    def forward(self, *inputs, **kw):
        raise NotImplementedError


class ModelWithNeighborhood(MainModelAbstract):
    """
    Adds tools to work with neighborhoods.
    """
    def __init__(self, neighborhood_type: str = None,
                 neighborhood_radius=None, **kw):
        """
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
        self.neighborhood_points = prepare_neighborhood_vectors(
            neighborhood_type, neighborhood_radius)
        self.nb_neighbors = len(self.neighborhood_points) if \
            self.neighborhood_points is not None else 0

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
            '--neighborhood_radius', type=Union[int, float, List[float]],
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
                    "To use an embedding class, you must provide its output "
                    "size")
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
        self.model_uses_dirs = True

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

    @property
    def params_for_json_prints(self):
        p = super().params_for_json_prints
        p.update({
            'prev_dirs_embedding':
                self.prev_dirs_embedding.params if
                self.prev_dirs_embedding else None,
        })
        return p

    def normalize_and_embed_previous_dirs(
            self, dirs: Union[list, torch.tensor],
            unpack_results: bool = True, point_idx=None) \
            -> Union[PackedSequence, List[torch.tensor], None]:
        """
        Runs the self.prev_dirs_embedding layer, if instantiated, and returns
        the model's output. Else, returns the data as is. Should be used in
        your forward method.

        Params
        ------
        dirs: list or tensor
            Batch all streamline directions If it is a tensor, it should be of
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
        n_prev_dirs_embedded: Union[List[Tensor], PackedSequence, None]
            The previous dirs, as list of tensors (or as PackedSequence if
            unpack_result is False). Returns None if nb_previous_dirs is 0.
        """
        if self.nb_previous_dirs == 0:
            return None

        if self.normalize_prev_dirs:
            dirs = normalize_directions(dirs)

        # Formatting the n previous dirs for all points.
        n_prev_dirs = self.format_previous_dirs(dirs, point_idx=point_idx)

        if not point_idx:
            # Not keeping the last point: only useful to get the last
            # direction (ex, last target), but won't be used as an input.
            n_prev_dirs = [s[:-1] for s in n_prev_dirs]

        if self.prev_dirs_embedding is None:
            return n_prev_dirs

        # Embedding, if asked
        is_list = isinstance(n_prev_dirs, list)
        n_prev_dirs_packed = None
        if is_list:
            # We could loop on all lists and embed each.
            # Probably faster to pack result, run model once on all points
            # and unpack later.
            n_prev_dirs_packed = pack_sequence(n_prev_dirs,
                                               enforce_sorted=False)

            n_prev_dirs = n_prev_dirs_packed.data
            n_prev_dirs.to(self.device)

        # Result is a tensor
        n_prev_dirs_embedded = self.prev_dirs_embedding(n_prev_dirs)

        if is_list:
            # Packing back to correctly unpack.
            n_prev_dirs_embedded = \
                PackedSequence(n_prev_dirs_embedded,
                               n_prev_dirs_packed.batch_sizes,
                               n_prev_dirs_packed.sorted_indices,
                               n_prev_dirs_packed.unsorted_indices)

            if unpack_results:
                n_prev_dirs_embedded = unpack_sequence(n_prev_dirs_embedded)

        return n_prev_dirs_embedded

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

    def forward(self, inputs, target_dirs: List[torch.tensor], **kw):
        """
        Params
        ------
        inputs: Any
            Batch of inputs.
            [nb_points, nb_features].
        target_dirs: List[torch.tensor]
            Directions computed from the streamlines. Not normalized yet.
        """
        # Hints : Should start with:

        # n_prev_dirs_embedded = self.normalize_and_embed_previous_dirs(
        #       target_dirs)

        # Concatenate with inputs
        # Ex: inputs = torch.cat((inputs, n_prev_dirs_embedded), dim=-1)
        raise NotImplementedError


<<<<<<< HEAD
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


class MainModelForTracking(MainModelAbstract):
    """
    Adding typical options for models intended for learning to track.
    - Last layer should be a direction getter.
    - Added option to save the estimated outputs after a batch to compare
      with targets (in the case of regression).
    """
    def __init__(self, dg_key: str = 'cosine-regression',
                 dg_args: dict = None, normalize_targets: bool = True,
                 allow_saving_estimated_outputs: bool = False, **kw):
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
        normalize_targets: bool
            If true, target streamline's directions vectors are normalized
            (norm=1). If the step size is fixed, it shouldn't make any
            difference. If streamlines are compressed, in theory you should
            normalize, but you could hope that not normalizing could give back
            to the algorithm a sense of distance between points.
            If true and the dg_key is a regression model, then, output
            directions are also normalized too. Default: True.
        allow_saving_estimated_outputs: bool
            If true, when trainer calls the forward method with
            save_estimated_outputs = True, we will save the estimated outputs
            as a SFT along with the targets for that batch to allow comparison.
            Can only be set to true if the model is a regression model.
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

        # About the targets and outputs
        self.normalize_targets = normalize_targets
        self.normalize_outputs = False
        if normalize_targets and 'regression' in self.dg_key:
            self.normalize_outputs = True

        # Verify if outputs can be saved for visualisation during training
        self.allow_saving_estimated_outputs = allow_saving_estimated_outputs
        if 'regression' not in self.dg_key and allow_saving_estimated_outputs:
            logging.warning(
                "You wanted to allow your model to save estimated outputs, "
                "but it is not a regression model. Ignoring.")
            self.allow_saving_estimated_outputs = False

        # To tell our trainer what to send to the forward / loss methods.
        if self.allow_saving_estimated_outputs:
            self.model_uses_streamlines = True  # Else, false like in super.
        self.model_uses_dirs = True

    def instantiate_direction_getter(self, dg_input_size):
        direction_getter_cls = keys_to_direction_getters[self.dg_key]
        self.direction_getter = direction_getter_cls(
            dg_input_size, **self.dg_args)

    @staticmethod
    def add_args_tracking_model(p):
        p.add_argument(
            '--normalize_directions', action='store_true',
            help="If true, directions will be normalized, both during "
                 "tracking (usually, we normalize. But by not normalizing and "
                 "working with compressed streamlines, you could hope your "
                 "model will gain a sense of distance) and during training "
                 "(if you train a regression model).")
        add_direction_getter_args(p)

    @property
    def params_for_checkpoint(self):
        p = super().params_for_checkpoint
        p.update({
            'dg_key': self.dg_key,
            'dg_args': self.dg_args,
            'normalize_targets': self.normalize_targets,
            'allow_saving_estimated_outputs': self.allow_saving_estimated_outputs
        })
        return p

    @property
    def params_for_json_prints(self):
        params = super().params_for_json_prints
        params.update({
            'direction_getter': self.direction_getter.params,
        })
        return params

    def _save_estimated_output(self, target_streamlines, target_dirs,
                               model_outputs, ref, estimated_output_path,
                               space, origin):
        """
        Converts targets and model outputs to tractograms and saves them.
        """
        sft_target = StatefulTractogram(target_streamlines, ref,
                                        space, origin)
        save_tractogram(sft_target,
                        os.path.join(estimated_output_path,
                                     'last_batch_targets.trk'))

        output_streamlines = self._format_output_to_streamlines(
            model_outputs, target_streamlines, target_dirs)
        sft_output = StatefulTractogram(output_streamlines, ref,
                                        space, origin)
        save_tractogram(sft_output,
                        os.path.join(estimated_output_path,
                                     'last_batch_estimated_outputs.trk'))

    def _format_output_to_streamlines(self, output_dirs, ref_streamlines,
                                      ref_dirs):
        """
        Depending on your model's output format, transform to streamlines
        format.

        Here is an example of use. Overwrite if it does not fit with your data.

        Params
        ------
        model_outputs: Any
            Your model's output.
        ref_streamlines: list
            The target streamlines, to use as reference.
        ref_dirs: list
            The target dirs.

        Returns
        -------
        streamlines: list
            The streamlines.
        """
        if not self.normalize_outputs:
            # We normalize them here to eventually give them the right length.
            output_dirs = normalize_directions(output_dirs)

        initial_lengths = [torch.linalg.norm(s, dim=-1, keepdim=True) for s in
                           ref_dirs]

        # First point is the same
        # Next points start from real streamline but advance in the output
        # direction instead. We don't know the step size (or compress use) so
        # giving them the same lengths as reference dirs.
        output_streamlines = [
            s[0].append(s[0:-1] + d * initial_lengths[i]) for
            i, s, d in enumerate(zip(ref_streamlines, output_dirs))]

        return output_streamlines

    def forward(self, inputs, target_dirs, target_streamlines=None,
                save_estimated_outputs: bool = False,
                ref=None, estimated_output_path: str = None,
                space: str = None, origin: str = None, **kw):
        """
        Params
        ------
        inputs: Any
        target_dirs: List[torch.tensor]
            Unnormalized directions.
        target_streamlines: List[torch.tensor],
            Batch of streamlines (only necessary to save estimated outputs,
            if asked).
        save_estimated_outputs: bool
            Trainer's opinion of if we should save the estimated outputs. Will
            only be done if self.allow_saving_estimated_outputs.
        ref: Any
            A reference to save the resulting SFT.
        estimated_output_path: str
            The saving path for the estimated results.
        space: str
            The training space (vox, voxmm, rasmm).
        origin: str
            The training origin (corner, center).
        """
        # Should end with:
        # model_outputs = self.direction_getter(last_layer_output)
        # if self.normalize_outputs:
        #     model_outputs = normalize_directions(model_outputs)
        #
        # if save_estimated_outputs and self.allow_saving_estimated_outputs:
        #    self._save_estimated_outputs(
        #        streamlines, dirs, model_outputs,
        #        ref, estimated_output_path, space, origin)
        raise NotImplementedError

<<<<<<< HEAD
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

    def compute_loss(self, model_outputs, target_dirs, target_streamlines=None,
                     **kw):
        # Should probably use:
        # if self.normalize_targets:
        #     target_dirs = normalize_directions(target_dirs)
        raise NotImplementedError

    def move_to(self, device):
        super().move_to(device)
        self.direction_getter.to(device)
