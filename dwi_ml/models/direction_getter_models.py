# -*- coding: utf-8 -*-
import logging
from math import ceil
from typing import Tuple, List, Union

import dipy.data
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal
from torch.nn import (CosineSimilarity, Dropout, Linear, ModuleList, ReLU,
                      KLDivLoss)
from torch.nn.modules.distance import PairwiseDistance

from dwi_ml.data.processing.streamlines.post_processing import \
    normalize_directions, compute_directions, compress_streamline_values, \
    weight_value_with_angle
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, convert_dirs_to_class
from dwi_ml.data.spheres import TorchSphere
from dwi_ml.models.utils.gaussians import independent_gaussian_log_prob
from dwi_ml.models.utils.fisher_von_mises import fisher_von_mises_log_prob

"""
The complete formulas and explanations are available in our doc:
https://dwi-ml.readthedocs.io/en/latest/model.html
"""


def init_2layer_fully_connected(input_size: int, output_size: int):
    """
    Defines a 2-layer fully connected network with sizes
    input_size   -->   input_size/2   -->   output_size
    """
    h1_size = ceil(input_size / 2)
    h1 = Linear(input_size, h1_size)
    h2 = Linear(h1_size, output_size)
    layers = ModuleList([h1, h2])

    return layers


def binary_cross_entropy_eos(learned_eos, target_eos, average_results=True):
    reduction = 'mean' if average_results else 'none'

    learned_eos = torch.sigmoid(learned_eos)
    losses_eos = torch.nn.functional.binary_cross_entropy(
        learned_eos, target_eos, reduction=reduction)
    return losses_eos


def _mean_and_weight(losses):
    # Mean:
    # Average on all time steps (all sequences) in batch
    # Keeping the gradients attached to allow backward propagation.
    mean = losses.mean()

    # STD: we would love to measure this to. But we can't average std values
    # of batches, even knowing n. Depends on the covariance between samples.
    # So this would not be useful for final epoch monitoring.

    # Also returning n. When running on batches, we want to compute the
    # mean of means of samples with different size, we need n.
    n = len(losses)

    return mean, n


class AbstractDirectionGetterModel(torch.nn.Module):
    """
    Default static class attribute, to be redefined by subclasses.

    Prepares the main functions. All models will be similar in the way that
    they all define layers. Then, we always apply self.loop_on_layers()

    input  -->  layer_i --> ReLu --> dropout -->  last_layer --> output
                  |                     |
                  -----------------------
    """
    def __init__(self, input_size: int, key: str,
                 supports_compressed_streamlines: bool, dropout: float = None,
                 compress_loss: bool = False, compress_eps: float = 1e-3,
                 weight_loss_with_angle: bool = False,
                 loss_description: str = '', add_eos: bool = False,
                 eos_weight: float = 1.0):
        """
        Parameters
        ----------
        input_size: int
            Should be computed directly. Probably the output size of the first
            layers of your main model.
        key: str
            The (children) class's key.
        supports_compressed_streamlines: bool
            Whether this model supports compressed streamlines.
        dropout: float
            Dropout rate. Usage depends on the child class.
        compress_loss: bool
            If set, compress the loss. This is used independently of the state
            of the streamlines received (compressed or resampled).
        compress_eps: float
            Compression threshold. As long as the angle is smaller than eps
            (in rad), the next points' loss are averaged together.
        weight_loss_with_angle: bool
            If set, weight loss with local angle. Can't be used together with
            compress_loss.
        loss_description: str
            Only meant to help users.
        add_eos: bool
            If true, child class should manage EOS.
        eos_weight: float
            If add_eos, proportion of the loss for EOS when it is calculated
            separately. ** Cannot be used with classification.
            Final loss will be: loss + eos_weight * eos.
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = None
        self.device = None
        self.add_eos = add_eos
        self.eos_weight = eos_weight
        self.compress_loss = compress_loss
        self.compress_eps = compress_eps
        self.weight_loss_with_angle = weight_loss_with_angle
        if self.compress_loss and self.weight_loss_with_angle:
            raise ValueError("We don't think it is a very good idea to use "
                             "option weight_loss_with_angle together "
                             "with compress_loss. They both serve the same "
                             "purpose.")

        # Info on this Direction Getter
        self.key = key
        self.supports_compressed_streamlines = supports_compressed_streamlines
        self.loss_description = loss_description

        # Preparing layers
        if dropout and (dropout < 0 or dropout > 1):
            raise ValueError('Dropout rate should be between 0 and 1.')
        self.dropout = dropout
        if self.dropout:
            self.dropout_sublayer = Dropout(self.dropout)
        else:
            self.dropout_sublayer = lambda x: x

        self.relu_sublayer = ReLU()

    def move_to(self, device):
        """
        Careful. Calling model.to(a_device) does not influence the self.device.
        Prefer this method for easier management.
        """
        self.to(device, non_blocking=True)
        self.device = device

    @property
    def params(self):
        """All parameters necessary to create again the same model. Will be
        used in the trainer, when saving the checkpoint state. Params here
        will be used to re-create the model when starting an experiment from
        checkpoint. You should be able to re-create an instance of your
        model with those params."""
        params = {
            'input_size': self.input_size,
            'dropout': self.dropout,
            'key': self.key,
            'add_eos': self.add_eos,
            'eos_weight': self.eos_weight,
            'compress_loss': self.compress_loss,
            'compress_eps': self.compress_eps,
            'weight_loss_with_angle': self.weight_loss_with_angle,
            'loss_description': self.loss_description
        }

        return params

    def loop_on_layers(self, inputs: Tensor, layers: ModuleList):
        """
        Apply a list of layers, using the ReLU activation function and dropout
        in-between layers.
        """
        layer_inputs = inputs

        for h in layers[:-1]:
            pre_act = h(layer_inputs)  # Pre-activations
            act = self.relu_sublayer(pre_act)  # Activations
            drop = self.dropout_sublayer(act)  # Dropout if necessary
            layer_inputs = drop

        # Apply last layer without activation function or dropout
        outputs = layers[-1](layer_inputs)
        return outputs

    def forward(self, inputs: Tensor):
        """
        Expecting a single tensor. Hint: either concatenate all streamlines'
        tensors, or loop on streamlines.
        """
        # Will be implemented by each class
        raise NotImplementedError

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Returns: List[Tensor], the directions
        """
        return target_dirs

    def compute_loss(self, outputs: List[Tensor],
                     target_streamlines: List[Tensor], average_results=True):
        """
        Parameters
        ----------
        outputs: List[Tensor]
            Your model's outputs
        target_streamlines: List[Tensor]
            The streamlines. Directions will be computed and formatted based
            on child class requirements.
        average_results: bool
            If true, returns the average over all values.

        Returns
        -------
        If compress_loss or average_results:
            Tuple(tensor, n)
                The average loss and the n points averaged.
        Else:
            List[Tensor]
                The loss for each point in each streamline.
        """
        if self.compress_loss and not average_results:
            raise ValueError("Current implementation of compress_loss does "
                             "not allow returning non-averaged loss.")

        # Compute directions
        target_dirs = compute_directions(target_streamlines)

        # For compress_loss and weight_with_angle: remember raw target dirs.
        # Also, do not average now, we will do our own averaging.
        target_dirs_copy = None
        tmp_average_results = average_results
        if self.weight_loss_with_angle or self.compress_loss:
            target_dirs_copy = [t.detach().clone() for t in target_dirs]
            tmp_average_results = False

        # Modify directions based on child model requirements.
        # Ex: Add eos label. Convert to classes. Etc.
        target_dirs = self._prepare_dirs_for_loss(target_dirs)
        lengths = [len(t) for t in target_dirs]

        # Stack and compute loss based on child model's loss definition.
        outputs, target_dirs = self.stack_batch(outputs, target_dirs)
        loss, n = self._compute_loss(outputs, target_dirs, tmp_average_results)

        # Finalize
        if self.weight_loss_with_angle:
            loss = list(torch.split(loss, lengths))
            if self.add_eos:
                eos_loss = [line_loss[-1] for line_loss in loss]
                loss = [line_loss[:-1] for line_loss in loss]
                loss = weight_value_with_angle(
                    values=loss, streamlines=None, dirs=target_dirs_copy)
                for i in range(len(loss)):
                    loss[i] = torch.hstack((loss[i], eos_loss[i]))
            else:
                loss = weight_value_with_angle(
                    values=loss, streamlines=None, dirs=target_dirs_copy)
            if not self.compress_loss:
                if average_results:
                    loss = torch.hstack(loss)
                    return torch.mean(loss), len(loss)
                return loss

        if self.compress_loss:
            loss = list(torch.split(loss, lengths))
            final_loss, final_n = compress_streamline_values(
                streamlines=None, dirs=target_dirs_copy, values=loss,
                compress_eps=self.compress_eps)
            logging.info("Converted {} data points into {} compressed data "
                         "points".format(sum(lengths), final_n))
            return final_loss, final_n
        elif average_results:
            return loss, n
        else:
            loss = list(torch.split(loss, lengths))
            return loss

    @staticmethod
    def stack_batch(outputs, target_dirs):
        target_dirs = torch.vstack(target_dirs)
        outputs = torch.vstack(outputs)
        return outputs, target_dirs

    def _compute_loss(
            self, outputs: Tensor, target_dirs: Tensor,
            average_results=True) -> Union[Tuple[Tensor, int], Tensor]:
        """
        Expecting a single tensor.

        Returns
        -------
        if average_results: Tuple
            mean_loss: Tensor
                Our direction getters' forward models output a single tensor,
                from concatenated streamlines.
            n: int, Total number of data points in this batch.
        else:
            losses: Tensor of shape (n, )
        """
        raise NotImplementedError

    def _sample_tracking_direction_prob(
            self, outputs, eos_stopping_thresh: Union[float, str]) -> Tensor:
        """
        Params:
        -------
        outputs: Any
            The output of the model after running its forward method.
        eos_stopping_thresh: float or 'max'

        Returns a direction per point, sampled following the model's
        distribution.
        """
        # Will be implemented by each class
        raise NotImplementedError

    def _get_tracking_direction_det(
            self, outputs, eos_stopping_thresh: Union[float, str]) -> Tensor:
        """
        Params:
        -------
        outputs: Any
            The output of the model after running its forward method.
        eos_stopping_thresh: float or 'max'

        Returns a direction per point, chosen deterministically.
        """
        # Will be implemented by each class
        raise NotImplementedError

    def get_tracking_directions(self, outputs, algo: str,
                                eos_stopping_thresh: Union[float, str]):
        """
        Parameters
        ----------
        outputs: Any
            The model's outputs
        algo: str
            Either 'det' or 'prob'
        eos_stopping_thresh: float or 'max'

        Returns
        -------
        next_dirs: torch.Tensor
            A tensor of shape [n, 3] with the next direction for each output.
        """
        if algo == 'det':
            next_dirs = self._get_tracking_direction_det(
                outputs, eos_stopping_thresh)
        else:
            next_dirs = self._sample_tracking_direction_prob(
                outputs, eos_stopping_thresh)
        return next_dirs


class AbstractRegressionDG(AbstractDirectionGetterModel):
    """
    Regression model.

    We use fully-connected (linear) network converting the outputs
    (at every step) to a 3D vector. Will use super to loop on layers, where
    layers are:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = 3

    EOS usage: uses a 4th dimension to targets to learn the SOS label.
    """
    def __init__(self, normalize_targets: float = None,
                 normalize_outputs: float = False, **kwargs):
        """
        normalize_targets: float
            Value to which to normalize targets when computing loss.
            Influences the result in the L2 case. No influence on the cosine
            loss. Default: 0 (no normalization).
        normalize_outputs: float
            Value to which to normalize the learned direction.
            Default: 0 (no normalization).
        """
        super().__init__(**kwargs)

        self.output_size = 3  # x, y, z direction.
        if self.add_eos:
            self.output_size = 4  # + EOS choice.
        self.layers = init_2layer_fully_connected(self.input_size,
                                                  self.output_size)
        self.normalize_targets = normalize_targets
        self.normalize_outputs = normalize_outputs

    @property
    def params(self):
        p = super().params
        p.update({
            'normalize_targets': self.normalize_targets,
            'normalize_outputs': self.normalize_outputs,
        })
        return p

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        output = self.loop_on_layers(inputs, self.layers)

        if self.normalize_outputs:
            output = normalize_directions(output) * self.normalize_outputs
        return output

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Should be called before _compute_loss, before concatenating your
        streamlines.

        Returns: list[Tensors], the directions.
        """
        # Need to normalize before adding EOS labels (dir = 0,0,0)
        if self.normalize_targets is not None:
            target_dirs = normalize_directions(target_dirs,
                                               new_norm=self.normalize_targets)
        return add_label_as_last_dim(target_dirs, add_sos=False,
                                     add_eos=self.add_eos)

    def _compute_loss(self, learned_directions: Tensor, target_dirs: Tensor,
                      average_results=True):

        # 1. Main loss:
        loss_dirs = self._compute_loss_dir(learned_directions[:, 0:3],
                                           target_dirs[:, 0:3])

        n = 1
        if average_results:
            loss_dirs, n = _mean_and_weight(loss_dirs)

        # 2. EOS loss:
        if self.add_eos:
            # Using last dim as EOS
            learned_eos = learned_directions[:, 3]
            target_eos = target_dirs[:, 3]

            # Binary cross-entropy
            loss_eos = binary_cross_entropy_eos(learned_eos, target_eos,
                                                average_results)

            return loss_dirs + self.eos_weight * loss_eos, n
        else:
            return loss_dirs, n

    def _sample_tracking_direction_prob(self, *arg, **kwargs):
        raise ValueError("Regression models do not support probabilistic "
                         "tractography.")

    def _get_tracking_direction_det(
            self, model_outputs: Tensor, eos_stopping_thresh: float):
        """
        In this case, the output is directly a 3D direction, so we can use it
        as is for the tracking.
        """
        if self.add_eos:
            eos_prob = torch.sigmoid(model_outputs[:, -1])
            eos_prob = torch.gt(eos_prob, eos_stopping_thresh)
            return torch.masked_fill(
                model_outputs[:, 0:3], eos_prob[:, None], torch.nan)
        else:
            return model_outputs


class CosineRegressionDG(AbstractRegressionDG):
    """
    Regression model. Loss = negative cosine similarity.
    """
    def __init__(self, **kwargs):
        super().__init__(key='cosine-regression',
                         supports_compressed_streamlines=False,
                         loss_description='negative cosine similarity',
                         **kwargs)

        # Loss will be applied on the last dimension.
        self.cos_loss = CosineSimilarity(dim=-1)

    def _compute_loss_dir(self, learned_directions: Tensor,
                          target_directions: Tensor):
        # A reminder: a cosine of -1 = aligned but wrong direction!
        #             a cosine of 0 = 90 degrees apart.
        #             a cosine of 1 = small angle
        # Thus we aim for a big cosine (maximize)! We minimize -cosine.
        # Best loss = -1. Adding 1. Best loss = 0.
        losses = 1.0 - self.cos_loss(learned_directions, target_directions)
        return losses


class L2RegressionDG(AbstractRegressionDG):
    """
    Regression model.

    Loss = Pairwise distance = p_root(sum(|x_i|^P))
    * If sequence: averaged on time steps and sequences.
    """
    def __init__(self, **kwargs):
        # L2 Distance loss supports compressed streamlines
        super().__init__(key='l2-regression',
                         supports_compressed_streamlines=True,
                         loss_description="torch's pairwise distance",
                         **kwargs)

        # Loss will be applied on the last dimension, by default in
        # PairWiseDistance
        self.l2_loss = PairwiseDistance(p=2)

        # Range of the L2: depends on the scaling of directions.
        # Ex if directions are normalized:
        # 180 degree = [0, 0, 1] and [0, 0, -1]:
        #            = sqrt(0 + 0 + 2^2) = sqrt(4) = 2
        #                 = sqrt((2*step_size)^2)
        #                 = sqrt(4*step_size^2)
        #                 = 2 * step_size.
        # 90 degree  = [0, 0, 1] and [0, 1, 0]:
        #            = sqrt(0 + 1^2 + 1^2) = sqrt(2)
        #                 = sqrt(2*step_size^2)
        #                 = sqrt(2) * step_size
        # Generally, step size is smaller than one voxel, so smaller than 2.
        # Minimum = 0.

    def _compute_loss_dir(self, learned_directions: Tensor,
                          target_directions: Tensor):
        losses = self.l2_loss(learned_directions, target_directions)
        return losses


class CosPlusL2RegressionDG(AbstractRegressionDG):
    def __init__(self, **kwargs):
        super().__init__(key='cos-plus-l2-regression',
                         supports_compressed_streamlines=False,
                         loss_description='l2 + negative cosine similarity',
                         **kwargs)

        # Loss will be applied on the last dimension.
        self.cos_loss = CosineSimilarity(dim=-1)
        self.l2_loss = PairwiseDistance()

    def _compute_loss_dir(self, learned_directions: Tensor,
                          target_directions: Tensor):
        l2_losses = self.l2_loss(learned_directions, target_directions)
        cos_losses = 1.0 - self.cos_loss(learned_directions, target_directions)
        losses = l2_losses + cos_losses

        return losses


class AbstractSphereClassificationDG(AbstractDirectionGetterModel):
    """
    Classification model.

    Classes: Points on the sphere.

    Model: We use fully-connected (linear) network converting the outputs
    (at every step) to a nD vector where n is the number of classes.
    We will use super to loop on layers, where layers are:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = n
    """
    def __init__(self, sphere: str = 'symmetric724', **kwargs):
        """
        sphere: str
            An choice of dipy's Sphere.
        """
        super().__init__(**kwargs)

        # Classes
        self.sphere_name = sphere
        sphere = dipy.data.get_sphere(sphere)
        self.torch_sphere = TorchSphere(sphere)
        self.output_size = sphere.vertices.shape[0]   # nb_classes

        # EOS
        if self.eos_weight != 1.0:
            raise NotImplementedError(
                "Current EOS computation when using classification cannot "
                "be used with eos_weight: Loss is computed all at once on "
                "all classes, including EOS.")

        if self.add_eos:
            self.output_size += 1
            self.eos_class_idx = self.output_size - 1  # Last class. Idx -1.
            # During tracking: tried choosing EOS if it's the class with max
            # logit. Seems to be too intense. Could be chosen even if
            # probability is very low (ex: if all 725 classes with sphere
            # symmetric724 are equal, probability of 0.001). Relaxing by
            # choosing it if its probability is more than 0.5. In papers: we
            # have seen "if more than the sum of all others". Not implemented
            # here, probably even more strict.

        self.layers = init_2layer_fully_connected(self.input_size,
                                                  self.output_size)

    def move_to(self, device):
        super().move_to(device)
        self.torch_sphere.move_to(device)

    @property
    def params(self):
        params = super().params

        params.update({
            'sphere': self.sphere_name,
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        logits = self.loop_on_layers(inputs, self.layers)
        return logits

    def _sample_tracking_direction_prob(
            self, logits_per_class: Tensor, eos_stopping_thresh):
        """
        Sample a tracking direction on the sphere from the predicted class
        logits (=probabilities). If class = EOS, direction is [NaN, NaN, NaN].
        """
        # Sample a direction on the sphere
        idx = Categorical(logits=logits_per_class).sample()

        return idx

    def _get_tracking_direction_det(
            self, logits_per_class: Tensor, eos_stopping_thresh):
        """
        Get the predicted class with highest logits (=probabilities).
        If class = EOS, direction is [NaN, NaN, NaN].
        """
        idx = logits_per_class.argmax(dim=1)
        return idx

    def get_tracking_directions(self, logits_per_class, algo: str,
                                eos_stopping_thresh: Union[float, str]):
        if not self.add_eos:
            idx = super().get_tracking_directions(
                logits_per_class, algo, eos_stopping_thresh)
            return self.torch_sphere.vertices[idx]
        else:
            # Get directions
            idx = super().get_tracking_directions(
                 logits_per_class[:, :-1], algo, eos_stopping_thresh)

            # But stop if EOS is bigger.
            if eos_stopping_thresh == 'max':
                # No need for the softmax: max will stay max.
                eos_chosen = logits_per_class[:, -1] > logits_per_class[:, idx]
            else:
                # In all our classification models, a softmax was used when
                # computing loss (or a log-softmax). Creating probabilities the
                # same way here.
                eos_probs = torch.softmax(logits_per_class, dim=-1)[:, -1]
                eos_chosen = eos_probs > eos_stopping_thresh
            idx[eos_chosen] = self.eos_class_idx
            return self.invalid_or_vertice(idx)

    def invalid_or_vertice(self, idx):
        stop = idx == self.eos_class_idx
        # Faking any vertex for best speed. Filling to nan after.
        idx[stop] = 0
        return self.torch_sphere.vertices[idx].masked_fill(
            stop[:, None], torch.nan)


class SphereClassificationDG(AbstractSphereClassificationDG):
    """
    Loss = negative log-likelihood.
    """
    def __init__(self, **kwargs):
        super().__init__(key='sphere-classification',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood',
                         **kwargs)

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Finds the closest class for each target direction.

        returns: List[Tensor], the index for each target.
        """
        target_idx = convert_dirs_to_class(
            target_dirs, self.torch_sphere, smooth_labels=False,
            add_sos=False, add_eos=self.add_eos, to_one_hot=False)
        return target_idx

    @staticmethod
    def stack_batch(outputs, target_dirs):
        # Formatted targets are a list of class index. Using hstack
        target_dirs = torch.hstack(target_dirs)
        # However, outputs are already a 'one-hot' vector per point (i.e. 2D).
        # Using vstack.
        outputs = torch.vstack(outputs)
        return outputs, target_dirs

    def _compute_loss(self, logits_per_class: Tensor, targets_idx: Tensor,
                      average_results=True):
        """
        Compute the negative log-likelihood for the targets using the
        model's logits.

        logits_per_class: Tensor of shape [nb_points, nb_class]
        targets_idx: Tensor of shape[nb_points], the target class indices.
        """
        # Create an official probability distribution from the logits
        # (i.e. applies a log-softmax).
        # By default, done on the last dim (the classes, for each point).
        learned_distribution = Categorical(logits=logits_per_class)

        # Target is an index for each point.

        # Get the logit at target's index,
        # Gets values on the first dimension (one target per point).
        nll_losses = -learned_distribution.log_prob(targets_idx)

        if average_results:
            return _mean_and_weight(nll_losses)
        else:
            n = 1
            return nll_losses, n


class SmoothSphereClassificationDG(AbstractSphereClassificationDG):
    """
    Loss = KL divergence
    """
    def __init__(self, **kwargs):
        """
        smooth_labels: bool
            If true, applies a Gaussian on the labels, as done in Deeptract.
        """
        super().__init__(key='smooth-sphere-classification',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood',
                         **kwargs)

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Finds the closest class for each target direction.

        returns:
        List[Tensor]: the one-hot distribution of each target.
        """
        target_idx = convert_dirs_to_class(
            target_dirs, self.torch_sphere, smooth_labels=True,
            add_sos=False, add_eos=self.add_eos, to_one_hot=True)

        return target_idx

    def _compute_loss(self, logits_per_class: Tensor, targets_probs: Tensor,
                      average_results=True):
        """
        Compute the negative log-likelihood for the targets using the
        model's logits.

        logits_per_class: Tensor of shape [nb_points, nb_class]
        targets_probs: Tensor of shape [nb_points, nb_class], the class indices
            as one hot vectors.
        """
        # Choice 1.
        # Careful with the order. Not the same as KLDivLoss
        # Create an official probability distribution from the logits
        # (i.e. applies a log_softmax first). What they actually do is:
        # self.logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        # which is mathematically the same.
        # By default, done on the last dim (the classes, for each point).
        # learned_distribution = Categorical(logits=logits_per_class)
        # target_distribution = Categorical(probs=targets_probs)
        # nll_losses = kl_divergence(target_distribution, learned_distribution)

        # Choice 2. kl_div and KLDivLoss (both equivalent but kl_div is
        # buggy: reduction is supposed to be a str but if I send 'none', it
        # says that it expects an int.)
        # Gives the same result as above, but averaged instead of summed.
        # The real definition is integral (i.e. sum). Typically, for our
        # data (724 classes), that's a big difference: from values ~7 to values
        # around 0.04. Nicer for visu with sum.
        # So, avoiding torch's 'mean' reduction; reducing ourselves.

        # Making sure our logits really are logits.
        logits_per_class = torch.log_softmax(logits_per_class, dim=-1)
        # Our targets are already probabilities after prepare_targets_for_loss.
        # Else, we could use:
        # targets_probs = torch.softmax(targets_probs, dim=-1)

        # Integral over classes per point.
        kl_loss = KLDivLoss(reduction='none', log_target=False)
        nll_losses = torch.sum(kl_loss(logits_per_class, targets_probs),
                               dim=-1)

        if average_results:
            return _mean_and_weight(nll_losses)
        else:
            return nll_losses


class SingleGaussianDG(AbstractDirectionGetterModel):
    """
    Regression model. The output is not an x,y,z value but the learned
    parameters of the gaussian representing the local direction.

    Not to be counfounded with the 3D Gaussian representing a tensor (means
    would be 0,0,0, the origin). This Gaussian's means represent the most
    probable direction, and variances represent the incertainty.

    Model: 2-layer NN for the means + 2-layer NN for the variances.

    Loss: Negative log-likelihood. See comments below.

    This model is very sensible to batches. If one batch has data of high
    certainty, and the next, not, learned sigmas vary a lot. Gradients are
    typically very big.

    It is known in the literature that Gaussian models lead to unstable
    gradients: see https://openreview.net/pdf?id=hmuLHC5MrG (under review)

    ===========> WE SUGGEST TO :
                USE GRADIENT CLIPPING **AND** low learning rate.
    """
    def __init__(self, normalize_targets: float = None,
                 entropy_weight: float = 0.0,  **kwargs):
        # 3D gaussian supports compressed streamlines
        super().__init__(key='gaussian',
                         supports_compressed_streamlines=True,
                         loss_description='negative log-likelihood',
                         **kwargs)

        self.normalize_targets = normalize_targets
        self.entropy_weight = entropy_weight

        # Layers
        # 3 values as mean, 3 values as sigma
        # If EOS: Adding it to the mean layer. Could be separated.
        oneifeos = 1 if self.add_eos else 0
        self.layers_mean = init_2layer_fully_connected(self.input_size,
                                                       3 + oneifeos)
        self.layers_sigmas = init_2layer_fully_connected(self.input_size, 3)
        self.output_size = 6 + oneifeos

        # Computes the negative log-likelihood from the difference between the
        # distribution and each target.

        # Note about the expected range:
        #   - likelihood = The learned normal value at correct direction value.
        #      Normal (as any PDF; probability distribution function) =
        #      non-negative + integrates to 1. But each value can be > 1.
        #      ex: 1D Normal: see with sigma = 0.1. Maximal values go > 4.
        #          3D Normal: with sigmas = 0.01, maximal values to > 60 000
        #   - so log-likelihood: Not limited by an asymptote. Can range from
        #     log(0) (bad) to log(inf) (good) = -inf to inf.
        #   - so NLL: range [bad, good] = [inf, -inf].

        # To test values with various sigma values:
        #   Suppose the target is [0, 0, 1] and we learn correctly the params:
        #   means = torch.as_tensor([0, 0, 1]). sigma = 0.01
        #   d = MultivariateNormal(means,
        #                          covariance_matrix=(sigma**2)*torch.eye(3))
        #   The value for the loss (should be minimal!):
        #   print(-d.log_prob(means)) --> -11.1
        #   The value for the likelihood (should be maximal!):
        #   print(torch.exp(d.log_prob(means)))  --> 63,493.6

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Should be called before _compute_loss, before concatenating your
        streamlines.

        Returns: list[Tensors], the directions.
        """
        # Need to normalize before adding EOS labels (dir = 0,0,0)
        if self.normalize_targets is not None:
            target_dirs = normalize_directions(target_dirs,
                                               new_norm=self.normalize_targets)
        return add_label_as_last_dim(target_dirs, add_sos=False,
                                     add_eos=self.add_eos)

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigmas)
        sigmas = torch.exp(log_sigmas)

        return means, sigmas

    @staticmethod
    def stack_batch(outputs, target_dirs):
        target_dirs = torch.vstack(target_dirs)
        means = torch.vstack(outputs[0])
        sigmas = torch.vstack(outputs[1])

        return (means, sigmas), target_dirs

    def _compute_loss(self, learned_gaussian_params: Tuple[Tensor, Tensor],
                      target_dirs: Tensor, average_results=True):
        """
        Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.

        See the doc for explanation on the formulas:
        https://dwi-ml.readthedocs.io/en/latest/formulas.html
        """
        # 1. Main loss
        means, sigmas = learned_gaussian_params
        learned_eos = None
        if self.add_eos:
            learned_eos = means[:, -1]
            means = means[:, 0:3]

        # Create an official function-probability distribution from the means
        # and variances
        distribution = MultivariateNormal(
            means, covariance_matrix=torch.diag_embed(sigmas ** 2))
        nll_loss = -distribution.log_prob(target_dirs[:, 0:3])

        if self.entropy_weight > 0:
            # Trying to ensure that sigma values are not too small.
            # Entropy values range between 0 and log(K). 0 = high probability.
            # We want a high entropy / low certainty = we will minimize
            # -entropy.
            entropy = distribution.entropy()
            logging.info("Computing batch loss with sigma {}, entropy: {}"
                         .format(torch.mean(sigmas), torch.mean(entropy)))
            nll_loss = nll_loss - self.entropy_weight * entropy

        n = 1
        if average_results:
            nll_loss, n = _mean_and_weight(nll_loss)

        # 2. EOS loss:
        if self.add_eos:
            # Binary cross-entropy
            loss_eos = binary_cross_entropy_eos(learned_eos,
                                                target_dirs[:, -1],
                                                average_results)
            return nll_loss + self.eos_weight * loss_eos, n
        else:
            return nll_loss, n

    def _sample_tracking_direction_prob(
            self, learned_gaussian_params: Tuple[Tensor, Tensor],
            eos_stopping_thresh):
        """
        From the gaussian parameters, sample a direction.
        """
        if self.add_eos:
            raise NotImplementedError

        means, sigmas = learned_gaussian_params

        # Sample a final function in the chosen Gaussian
        # One direction per time step per sequence
        distribution = MultivariateNormal(
            means[:, 0:3], covariance_matrix=torch.diag_embed(sigmas ** 2))
        direction = distribution.sample()

        if self.add_eos:
            eos_prob = torch.sigmoid(means[:, -1])
            eos_prob = torch.gt(eos_prob, eos_stopping_thresh)
            return torch.masked_fill(
                direction, eos_prob[:, None], torch.nan)
        else:
            return direction

    def _get_tracking_direction_det(self, learned_gaussian_params: Tensor,
                                    eos_stopping_thresh):
        """
        Get the predicted class with highest logits (=probabilities).
        """
        # Returns the direction of the max of the Gaussian = the mean.
        # Not using sigma
        means, sigmas = learned_gaussian_params
        dirs = means[:, 0:3]

        if self.add_eos:
            eos_prob = torch.sigmoid(means[:, -1])
            eos_prob = torch.gt(eos_prob, eos_stopping_thresh)
            return torch.masked_fill(dirs, eos_prob[:, None], torch.nan)
        else:
            return dirs


class GaussianMixtureDG(AbstractDirectionGetterModel):
    """
    Regression model. The output is not an x,y,z value but the learned
    parameters of the distribution representing the local direction.

    Same as SingleGaussian but with more than one Gaussian. This should account
    for branching bundles, distributing probability across space at branching
    points.

    Model: (a 2-layer NN for the mean and a 2-layer NN for the sigma.) for
    each of N Gaussians.
    (Parameters:     N Gaussians * (1 mixture param + 3 means + 3 sigmas))

    Loss: Negative log-likelihood.
    """
    def __init__(self, nb_gaussians: int = 3, **kwargs):
        # 3D Gaussian mixture supports compressed streamlines
        super().__init__(key='gaussian-mixture',
                         supports_compressed_streamlines=True,
                         loss_description='negative log-likelihood',
                         **kwargs)

        if self.add_eos:
            raise NotImplementedError

        self.nb_gaussians = nb_gaussians

        # N Gaussians * (1 mixture param + 3 means + 3 sigmas)
        # (no correlations between each Gaussian)
        # Use separate *heads* for the mu and sigma, but concatenate gaussians

        self.layers_mixture = init_2layer_fully_connected(
            self.input_size, 1 * self.nb_gaussians)
        self.layers_mean = init_2layer_fully_connected(
            self.input_size, 3 * self.nb_gaussians)
        self.layers_sigmas = init_2layer_fully_connected(
            self.input_size, 3 * self.nb_gaussians)

        self.output_size = 7 * self.nb_gaussians
        # Loss will be defined in _compute_loss, using torch distribution

    @property
    def params(self):
        params = super().params
        params.update({
            'nb_gaussians': self.nb_gaussians,
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        mixture_logits = self.loop_on_layers(inputs, self.layers_mixture)

        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigma)
        sigmas = torch.exp(log_sigmas)

        return mixture_logits, means, sigmas

    @staticmethod
    def stack_batch(outputs, target_dirs):
        target_dirs = torch.vstack(target_dirs)
        mixture_logits = torch.vstack(outputs[0])
        means = torch.vstack(outputs[1])
        sigmas = torch.vstack(outputs[2])
        return (mixture_logits, means, sigmas), target_dirs

    def _compute_loss(
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor],
            target_dirs, average_results=True):
        """
        Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.

        See the doc for explanation on the formulas:
        https://dwi-ml.readthedocs.io/en/latest/formulas.html
        """
        # Shape : [batch_size*seq_len, n_gaussians, 3] or
        #         [batch_size, n_gaussians, 3]
        mixture_logits, means, sigmas = \
            self._get_gaussian_parameters(learned_gaussian_params)

        # Take softmax for the mixture parameters
        mixture_probs = torch.softmax(mixture_logits, dim=-1)

        gaussians_log_prob = independent_gaussian_log_prob(
            target_dirs[:, None, :], means, sigmas)

        nll_losses = -torch.logsumexp(mixture_probs.log() + gaussians_log_prob,
                                      dim=-1)

        if average_results:
            return _mean_and_weight(nll_losses)
        else:
            return nll_losses

    def _sample_tracking_direction_prob(
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor],
            eos_stopping_thresh):
        """
        From the gaussian mixture parameters, sample one of the gaussians
        using the mixture probabilities, then sample a direction from the
        selected gaussian.
        """
        if self.add_eos:
            raise NotImplementedError

        mixture_logits, means, sigmas = \
            self._get_gaussian_parameters(learned_gaussian_params)

        # Create probability distribution and sample a gaussian per point
        # (or per time step per sequence)
        mixture_distribution = Categorical(logits=mixture_logits)
        mixture_id = mixture_distribution.sample()

        # For each point in the batch (of concatenated sequences) take the mean
        # and sigma parameters. Note. Means and sigmas are of shape
        # [batch_size*seq_len, n_gaussians, 3] or [batch_size, n_gaussians, 3]
        component_means = means[:, mixture_id, :]
        component_sigmas = sigmas[:, mixture_id, :]

        # Sample a final function in the chosen Gaussian
        # One direction per timestep per sequence.
        component_distribution = MultivariateNormal(
            component_means,
            covariance_matrix=torch.diag_embed(component_sigmas ** 2))
        direction = component_distribution.sample()

        return direction

    def _get_tracking_direction_det(
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor],
            eos_stopping_thresh):
        if self.add_eos:
            raise NotImplementedError

        mixture_logits, means, sigmas = \
            self._get_gaussian_parameters(learned_gaussian_params)

        # mixture_logits: [batch_size, n_gaussians]
        best_gaussian = torch.argmax(mixture_logits, dim=1)
        chosen_means = means[:, best_gaussian, :]

        return chosen_means

    def _get_gaussian_parameters(
            self, gaussian_params: Tuple[Tensor, Tensor, Tensor]):
        """From the model output, extract the mixture parameters, the means
        and the sigmas, all reshaped according to the number of components
        and the dimensions (3D). i.e. [batch_size, n_gaussians] for the mixture
        logit and [batch_size, n_gaussians, 3] for the means and sigmas.
        """
        mixture_logits = gaussian_params[0]   # .squeeze(dim=1)
        means = gaussian_params[1].reshape((-1, self.nb_gaussians, 3))
        sigmas = gaussian_params[2].reshape((-1, self.nb_gaussians, 3))

        return mixture_logits, means, sigmas


class FisherVonMisesDG(AbstractDirectionGetterModel):
    """
    Regression model. The output is not an x,y,z value but the learned
    parameters of the distribution representing the local direction.

    This model provides probabilistic outputs using the Fisher - von Mises
    distribution [1][2], which resembles a gaussian on the sphere. As such,
    it does not require unit normalization when sampling, and should be more
    stable while training.

    We sample using rejection sampling defined in [3], implemented in [4].

    Parameters are mu and kappa. Larger kappa leads to a more concentrated
    cluster of points, similar to sigma for Gaussians.

    Ref:
    [1]: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    [2]: http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    [3]: Directional Statistics (Mardia and Jupp, 1999)
    [4]: https://github.com/jasonlaska/spherecluster
    """
    def __init__(self, **kwargs):
        super().__init__(key='fisher-von-mises',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood',
                         **kwargs)

        # Layers
        # 3 values as mean, 1 value as kappa
        # If EOS: Adding it to the mean layer. Could be separated.
        oneifeos = 1 if self.add_eos else 0
        self.layers_mean = init_2layer_fully_connected(self.input_size,
                                                       3 + oneifeos)
        self.layers_kappa = init_2layer_fully_connected(self.input_size, 1)

        self.output_size = 4
        # Loss will be defined in _compute_loss, using torch distribution

    def _prepare_dirs_for_loss(self, target_dirs: List[Tensor]):
        """
        Should be called before _compute_loss, before concatenating your
        streamlines.

        Returns: list[Tensors], the directions.
        """
        # Need to normalize before adding EOS labels (dir = 0,0,0)
        target_dirs = normalize_directions(target_dirs)
        return add_label_as_last_dim(target_dirs, add_sos=False,
                                     add_eos=self.add_eos)

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the inputs through the fully-connected layer.

        Returns
        -------
        mus : torch.Tensor with shape [batch_size x 3]
            The 3D coordinate of the mean.
        kappas : torch.Tensor with shape [batch_size x 1]
            The kappa concentration parameter.
        """
        mu = self.loop_on_layers(inputs, self.layers_mean)
        kappas = self.loop_on_layers(inputs, self.layers_kappa)

        # mean should be a unit vector for Fisher Von-Mises distribution
        # (Using [0:3] only; EOS value does not need to be normalized).
        # Simple code line raises an error: inplace operation
        # mu[0:3] = torch.nn.functional.normalize(mu[0:3], dim=-1)
        learned_eos = None
        if self.add_eos:
            learned_eos = mu[:, 3][:, None]
            mu = mu[:, 0:3]
        mu = torch.nn.functional.normalize(mu, dim=-1)
        if self.add_eos:
            mu = torch.hstack((mu, learned_eos))

        # Need to restrict kappa to a certain range, e.g. [0, 20]
        kappas = torch.sigmoid(kappas) * 20

        # Squeeze the trailing dim, the kappa parameter is a scalar
        kappas = kappas.squeeze(dim=-1)

        return mu, kappas

    @staticmethod
    def stack_batch(outputs, target_dirs):
        target_dirs = torch.vstack(target_dirs)
        mu = torch.vstack(outputs[0])
        kappa = torch.hstack(outputs[1])  # Not vstack: they are vectors
        return (mu, kappa), target_dirs

    def _compute_loss(self, learned_fisher_params: Tuple[Tensor, Tensor],
                      target_dirs, average_results=True):
        """Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.

        See the doc for explanation on the formulas:
        https://dwi-ml.readthedocs.io/en/latest/formulas.html
        """
        # mu.shape : [all_point, 4]. 3 first values are x, y, z. Last is EOS.
        mu, kappa = learned_fisher_params
        learned_eos = None
        if self.add_eos:
            learned_eos = mu[:, 3]
            mu = mu[:, 0:3]

        # 1. Main loss
        # Note. Mu was already normalized through the forward method.
        log_prob = fisher_von_mises_log_prob(mu, kappa, target_dirs[:, 0:3])
        nll_loss = -log_prob

        n = 1
        if average_results:
            nll_loss, n = _mean_and_weight(nll_loss)

        # 2. EOS loss:
        if self.add_eos:
            # Binary cross-entropy
            loss_eos = binary_cross_entropy_eos(learned_eos,
                                                target_dirs[:, -1],
                                                average_results)
            return nll_loss + self.eos_weight * loss_eos, n
        else:
            return nll_loss, n

    def _sample_tracking_direction_prob(
            self, learned_fisher_params: Tuple[Tensor, Tensor],
            eos_stopping_thresh):
        """Sample directions from a fisher von mises distribution.
        """
        if self.add_eos:
            raise NotImplementedError

        # mu.shape : [flattened_sequences, 3]
        mus, kappas = learned_fisher_params

        n_samples = mus.shape[0]

        # Apply squeeze, everything should be single-step sequences
        mus = mus.squeeze(dim=1).cpu().numpy()
        kappas = kappas.squeeze(dim=1).cpu().numpy()

        result = np.zeros((n_samples, 3))
        for i, mu, kappa in zip(range(n_samples), mus, kappas):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(kappa)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(mu)

            # compute new point
            result[i, :] = v * np.sqrt(1. - w ** 2) + w * mu

        direction = torch.as_tensor(result, dtype=torch.float32)

        return direction

    def _get_tracking_direction_det(self, learned_fisher_params: Tensor,
                                    eos_stopping_thresh):
        """
        Get the predicted class with highest logits (=probabilities).
        """
        # Returns the direction of the max of the Gaussian = the mean.
        # Not using sigma
        mus, kappas = learned_fisher_params
        dirs = mus[:, 0:3]

        if self.add_eos:
            eos_prob = torch.sigmoid(mus[:, -1])
            eos_prob = torch.gt(eos_prob, eos_stopping_thresh)
            return torch.masked_fill(dirs, eos_prob[:, None], torch.nan)
        else:
            return dirs

    @staticmethod
    def _sample_weight(kappa):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        b = 2 / (np.sqrt(4. * kappa ** 2 + 4.) + 2 * kappa)
        x = (1. - b) / (1. + b)
        c = kappa * x + 2 * np.log(1 - x ** 2)

        while True:
            z = np.random.beta(1., 1.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + 2 * np.log(1. - x * w) - c >= np.log(u):
                return w

    @staticmethod
    def _sample_orthonormal_to(mu):
        """Sample point on sphere orthogonal to mu."""
        v = np.random.randn(mu.shape[0])
        proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)


class FisherVonMisesMixtureDG(AbstractDirectionGetterModel):
    """
    Compared to the version with 1 disbstribution, we now have an additional
    weight parameter alpha.
    """
    def __init__(self, n_cluster: int = 3, **kwargs):
        super().__init__(key='fisher-von-mises-mixture',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood',
                         **kwargs)

        self.n_cluster = n_cluster
        self.output_size = 5 * n_cluster

        # toDO
        raise NotImplementedError

    @property
    def params(self):
        params = super().params
        params.update({
            'n_cluster': self.n_cluster
        })
        return params

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _compute_loss(self, outputs: Tuple[Tensor, Tensor],
                      target_dirs: Tensor, average_results=True):
        raise NotImplementedError

    def _sample_tracking_direction_prob(self, outputs: Tuple[Tensor, Tensor],
                                        eos_stopping_thresh):
        raise NotImplementedError

    def _get_tracking_direction_det(self, learned_fisher_params: Tensor,
                                    eos_stopping_thresh):
        raise NotImplementedError


keys_to_direction_getters = \
    {'cosine-regression': CosineRegressionDG,
     'l2-regression': L2RegressionDG,
     'cosine-plus-l2-regression': CosPlusL2RegressionDG,
     'sphere-classification': SphereClassificationDG,
     'smooth-sphere-classification': SmoothSphereClassificationDG,
     'gaussian': SingleGaussianDG,
     'gaussian-mixture': GaussianMixtureDG,
     'fisher-von-mises': FisherVonMisesDG,
     'fisher-von-mises-mixture': FisherVonMisesMixtureDG}
