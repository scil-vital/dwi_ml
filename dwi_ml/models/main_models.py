# -*- coding: utf-8 -*-
import json
import logging
import os
import shutil
from typing import Union, Iterable

import torch

from dwi_ml.data.processing.space.neighborhood import (
    extend_coordinates_with_neighborhood,
    prepare_neighborhood_information)
from dwi_ml.data.processing.volume.interpolation import (
    torch_trilinear_interpolation)


class ModelAbstract(torch.nn.Module):
    """
    To be used for all sub-models (ex, layers in a main model).
    """

    def __init__(self):
        super().__init__()
        self.log = logging.getLogger()  # Gets the root

    @property
    def params(self):
        """All parameters necessary to create again the same model"""
        return {}

    def set_log(self, log: logging.Logger):
        """Possibility to pass a tqdm-compatible logger in case the dataloader
        is iterated through a tqdm progress bar. Note that, of course, log
        outputs will be confusing, particularly in debug mode, considering
        that the dataloader may use more than one method in parallel."""
        self.log = log


class MainModelAbstract(ModelAbstract):
    """
    To be used for all models that will be trained. Defines the way to save
    the model.

    It should also define a forward() method.
    """
    def __init__(self, experiment_name='my_model'):
        super().__init__()
        self.experiment_name = experiment_name
        self.best_model_state = None

    @property
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        return {
            'experiment_name': self.experiment_name
        }

    @classmethod
    def init_from_checkpoint(cls, **params):
        model = cls(**params)
        return model

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
        name = os.path.join(model_dir, "parametersr.json")
        with open(name, 'w') as json_file:
            json_file.write(json.dumps(self.params, indent=4,
                                       separators=(',', ': ')))

        # Save model
        torch.save(self.best_model_state,
                   os.path.join(model_dir, "best_model_state.pkl"))

        if to_remove:
            shutil.rmtree(to_remove)

    def compute_loss(self, outputs, targets):
        raise NotImplementedError


# One example of model
class MainModelAbstractNeighborsPreviousDirs(MainModelAbstract):
    def __init__(self, experiment_name, nb_features, input_group_name,
                 nb_previous_dirs, neighborhood_type: Union[str, None],
                 neighborhood_radius: Union[int, float, Iterable[float], None]
                 ):
        """
        Parameters
        ----------
        nb_previous_dirs : int
            If set, concatenate the n previous streamline directions as input.
            [0].
        neighborhood_type: str
            The type of neighborhood to add. One of 'axes', 'grid' or None. If
            None, don't add any. See
            dwi_ml.data.processing.space.Neighborhood for more information.
        neighborhood_radius : Union[int, float, Iterable[float]]
            Add neighborhood points at the given distance (in voxels) in each
            direction (nb_neighborhood_axes). (Can be none)
                - For a grid neighborhood: type must be int.
                - For an axes neighborhood: type must be float. If it is an
                iterable of floats, we will use a multi-radius neighborhood.
        """
        super().__init__(experiment_name)

        # Preparing the neighborhood
        self.neighborhood_type = neighborhood_type
        self.neighborhood_radius = neighborhood_radius
        self.neighborhood_points = self._prepare_neighborhood()

        # This should be one of the dataset's volume_groups.
        self.input_group_name = input_group_name
        self.nb_previous_dirs = nb_previous_dirs

        self.nb_features = nb_features
        self.input_size = nb_features * (len(self.neighborhood_points) + 1)

    def _prepare_neighborhood(self):
        if self.neighborhood_type is None:
            if self.neighborhood_radius:
                logging.warning(
                    "You have chosen not to add a neighborhood (value None), "
                    "but you have given a neighborhood radius. Discarded.")
            return []

        else:
            if self.neighborhood_radius is None:
                raise ValueError("You must provide neighborhood radius to add "
                                 "a neighborhood.")
            elif self.neighborhood_type not in ['axes', 'grid']:
                raise ValueError(
                    "Neighborhood type must be either 'axes', 'grid' or None, "
                    "but we received {}!".format(self.neighborhood_type))
            neighborhood_points = prepare_neighborhood_information(
                self.neighborhood_type, self.neighborhood_radius)

            return neighborhood_points

    @property
    def params(self):
        params = super().params

        # converting np.int64 to int to allow json dumps.
        params.update({
            'input_group_name': self.input_group_name,
            'nb_features': int(self.nb_features),
            'neighborhood_type': self.neighborhood_type,
            'neighborhood_radius': self.neighborhood_radius,
            'number of neighbors': len(self.neighborhood_points),
            'input_size': int(self.input_size),
            'nb_previous_dirs': self.nb_previous_dirs
        })

        return params

    def prepare_inputs(self, data_volume, coords, device):
        if self.neighborhood_type is not None:
            n_input_points = coords.shape[0]

            # Extend the coords array with the neighborhood coordinates
            coords = extend_coordinates_with_neighborhood(
                coords, self.neighborhood_points)

            # Interpolate signal for each (new) point
            coords_torch = torch.as_tensor(coords, dtype=torch.float,
                                           device=device)
            flat_subj_x_data, coords_clipped = \
                torch_trilinear_interpolation(data_volume, coords_torch)

            # Reshape signal into (n_points, new_nb_features)
            # DWI data features for each neighboor are contatenated.
            #    dwi_feat1_neig1  dwi_feat2_neig1 ...  dwi_featn_neighbm
            #  p1        .              .                    .
            #  p2        .              .                    .
            n_features = (flat_subj_x_data.shape[-1] *
                          (self.neighborhood_points.shape[0] + 1))
            subj_x_data = flat_subj_x_data.reshape(n_input_points, n_features)

        else:  # No neighborhood:
            # Interpolate signal for each point
            coords_torch = torch.as_tensor(coords, dtype=torch.float,
                                           device=device)
            subj_x_data, coords_clipped = \
                torch_trilinear_interpolation(data_volume.data, coords_torch)

        return subj_x_data, coords_torch, coords_clipped

    def prepare_previous_dirs(self, all_previous_dirs, device):
        """
        About device: see compute_inputs

        Returns:
        previous_dirs: list[tensor]
            A list of length nb_streamlines. Each tensor is of size
            [nb_time_step, nb_previous_dir x 3]; the n previous dirs at each
            point of the streamline. When previous dirs do not exist (ex,
            the 2nd previous dir at the first time step), value is 0.
        """
        # Compute previous directions
        if self.nb_previous_dirs == 0:
            return []

        # toDo See what to do when values do not exist. See discussion here.
        #  https://stats.stackexchange.com/questions/169887/classification-with-partially-unknown-data
        empty_coord = torch.zeros((1, 3), dtype=torch.float32, device=device)

        previous_dirs = \
            [torch.cat([torch.cat((empty_coord.repeat(min(len(s), i + 1), 1),
                                   s[:-(i + 1)]))
                        for i in range(self.nb_previous_dirs)],
                       dim=1)
             for s in all_previous_dirs]
        return previous_dirs

    def compute_loss(self, outputs, targets):
        raise NotImplementedError
