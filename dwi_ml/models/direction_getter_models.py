# -*- coding: utf-8 -*-
from math import ceil
from typing import Any, Tuple, List, Union

import dipy.data
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal
from torch.nn import (CosineSimilarity, Dropout, Linear, ModuleList, ReLU)
from torch.nn.modules.distance import PairwiseDistance

from dwi_ml.data.processing.streamlines.post_processing import \
    normalize_directions, compute_directions
from dwi_ml.data.processing.streamlines.sos_eos_management import \
    add_label_as_last_dim, convert_dirs_to_class
from dwi_ml.data.spheres import TorchSphere
from dwi_ml.models.utils.gaussians import \
    independent_gaussian_log_prob
from dwi_ml.models.utils.fisher_von_mises import fisher_von_mises_log_prob

"""
The complete formulas and explanations are available in our doc:
https://dwi-ml.readthedocs.io/en/latest/model.html


Expected types and shapes:

INPUTS:  Def: Here, we call 'input' the output of your experiment model.
              (Ex, from a RNN).
         Type: tensor
         Size:
             - Sequence models: [batch_size*seq_len, nb_features]
              where seq_len is the nb of time steps in the sequence.
              So the input sequences are concatenated.
             - Local models: [batch_size, nb_features]

OUTPUTS: Def: The model final outputs, corresponding to directions, i.e.
              (x,y,z) coordinates.
         Type: tensor
         Size:
             - Sequence models: [batch_size*seq_len, 3]
             - Local models: [batch_size, 3]

TARGETS: Def: The target values (real Y) for the batch
         Type: tensor
         Size:
             - Sequence models: [batch_size*seq_len, 3]
             - Local models: [batch_size, 3]
"""
eps = 1e-6


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


def _mean_and_weight(losses):
    # Mean:
    # Average on all timesteps (all sequences) in batch
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
    Default static class attribute, to be redefined by sub-classes.

    Prepares the main functions. All models will be similar in the way that
    they all define layers. Then, we always apply self.loop_on_layers()

    input  -->  layer_i --> ReLu --> dropout -->  last_layer --> output
                  |                     |
                  -----------------------
    """
    def __init__(self, input_size, dropout: float, key: str,
                 supports_compressed_streamlines: bool,
                 loss_description: str = ''):
        """
        Parameters
        ----------
        input_size: Any
            Should be computed directly. Probably the output size of the first
            layers of your main model.
        dropout: float
            Dropout rate.
        supports_compressed_streamlines: bool
            Whether this model supports compressed streamlines.
        loss_description: str
            Will be added to params to help user remember the direction getter
            version (when printing params).
        """
        super().__init__()

        self.input_size = input_size
        self.device = None

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
        }
        # Adding stuff for nicer print in logs, although not necessary to
        # instantiate a new model.
        params.update({
            'key': self.key,
            'loss': self.loss_description
        })

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

    def prepare_targets(self, target_streamlines: List[Tensor]):
        """
        Should be called before compute_loss, before concatenating your
        streamlines.
        """
        return compute_directions(target_streamlines)

    def compute_loss(self, outputs: Tensor, target_dirs: Tensor) -> Tuple:
        """
        Expecting a single tensor. Hint: either concatenate all streamlines'
        tensors, or loop on streamlines.

        Returns
        -------
        mean_loss: Tensor
            Our direction getters' forward models output a single tensor, from
            concatenated streamlines.
        n: int, Total number of data points in this batch.
        """
        # Will be implemented by each class
        # Should start with
        # target_dirs = self.prepare_targets(target_streamlines)
        raise NotImplementedError

    def _sample_tracking_direction_prob(self, outputs: Tensor) -> Tensor:
        """
        Params:
        -------
        outputs: Any
            The output of the model after running its forward method.

        Returns a direction, sampled following the model's distribution.
        """
        # Will be implemented by each class
        raise NotImplementedError

    def _get_tracking_direction_det(self, outputs: Tensor) -> Tensor:
        """
        Params:
        -------
        outputs: Any
            The output of the model after running its forward method.

        Returns a direction, chosen deterministically.
        """
        # Will be implemented by each class
        raise NotImplementedError

    def get_tracking_directions(self, outputs: Tensor, algo: str):
        """
        Parameters
        ----------
        outputs: Tensor
            The model's outputs
        algo: str
            Either 'det' or 'prob'

        Returns
        -------
        next_dirs: list
            A list of numpy arrays (one per streamline), each of size (1, 3):
            the three coordinates of the next direction's vector.
        """
        if algo == 'det':
            next_dirs = self._get_tracking_direction_det(outputs)
        else:
            next_dirs = self._sample_tracking_direction_prob(outputs)
        return next_dirs.detach()


class AbstractRegressionDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model.

    We use fully-connected (linear) network converting the outputs
    (at every step) to a 3D vector. Will use super to loop on layers, where
    layers are:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = 3
    """
    def __init__(self, input_size, dropout: float, key: str,
                 supports_compressed_streamlines: bool,
                 loss_description: str = '',
                 normalize_targets: float = False,
                 normalize_outputs: float = False,
                 add_eos_label: float = False):
        """
        normalize_targets: float
            Value to which to normalize targets when computing loss.
            Influences the result in the L2 case. No influence on the cosine
            loss. Default: 0 (no normalization).
        normalize_outputs: float
            Value to which to normalize the learned direction.
            Default: 0 (no normalization).
        """
        super().__init__(input_size, dropout, key,
                         supports_compressed_streamlines, loss_description)

        output_size = 3  # x, y, z direction.
        if add_eos_label:
            output_size = 4  # + EOS choice.
        self.layers = init_2layer_fully_connected(input_size, output_size)
        self.normalize_targets = normalize_targets
        self.normalize_outputs = normalize_outputs
        self.add_eos_label = add_eos_label

        # Verify the range for the regression loss
        self.scaling_factor_eos_loss = None

    @property
    def params(self):
        p = super().params
        p.update({
            'normalize_targets': self.normalize_targets,
            'normalize_outputs': self.normalize_outputs,
            'add_eos_label': self.add_eos_label
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

    def prepare_targets(self, target_streamlines: List[Tensor]):
        """
        Should be called before compute_loss, before concatenating your
        streamlines.
        """
        target_dirs = compute_directions(target_streamlines)
        return add_label_as_last_dim(target_dirs, add_sos=False,
                                     add_eos=self.add_eos_label)

    def compute_loss(self, learned_directions, target_dirs):

        # Divide direction loss and EOS loss.
        if self.add_eos_label:
            assert target_dirs.shape[-1] == 4, \
                "Don't forget to run prepare_targets before compute_loss."

            # Keeping last dim as EOS
            # Just zeros and ones (sum of booleans).
            # We then divide the score to have this loss similar to the
            # directions loss.
            losses_eos = torch.sum(
                torch.eq(learned_directions[-1, :], target_dirs[-1, :]))
            nb_streamlines = torch.sum(target_dirs[-1, :])
            losses_eos = losses_eos / nb_streamlines * self.scaling_factor_eos_loss

            mean_loss_eos, _ = _mean_and_weight(losses_eos)
            learned_directions = learned_directions[:-1, :]
            target_dirs = target_dirs[:-1, :]
        else:
            mean_loss_eos = torch.as_tensor(0, device=self.device)

        # Now the main loss
        if self.normalize_targets:
            target_dirs = normalize_directions(target_dirs) * self.normalize_targets

        losses_dirs = self._compute_loss(learned_directions, target_dirs)
        mean_loss_dir, n = _mean_and_weight(losses_dirs)

        return (mean_loss_dir, mean_loss_eos), n

    def _sample_tracking_direction_prob(self, model_outputs: Tensor):
        raise ValueError("Regression models do not support probabilistic "
                         "tractography.")

    def _get_tracking_direction_det(self, model_outputs: Tensor):
        """
        In this case, the output is directly a 3D direction, so we can use it
        as is for the tracking.
        """
        return model_outputs


class CosineRegressionDirectionGetter(AbstractRegressionDirectionGetter):
    """
    Regression model.

    Loss = negative cosine similarity.
    * If sequence: averaged on time steps and sequences.
    """
    def __init__(self, input_size: int, dropout: float = None,
                 normalize_targets: float = False,
                 normalize_outputs: float = False,
                 add_eos_label: bool = False):
        super().__init__(input_size, dropout,
                         key='cosine-regression',
                         supports_compressed_streamlines=False,
                         loss_description='negative cosine similarity',
                         normalize_targets=normalize_targets,
                         normalize_outputs=normalize_outputs,
                         add_eos_label=add_eos_label)

        # Loss will be applied on the last dimension.
        self.cos_loss = CosineSimilarity(dim=-1)

        # Range of the cosine: -1 and 1 = 2
        self.scaling_factor_eos_loss = 2

    def _compute_loss(self, learned_directions: Tensor,
                      target_directions: Tensor):
        # A reminder: a cosine of -1 = aligned but wrong direction!
        #             a cosine of 0 = 90 degrees apart.
        #             a cosine of 1 = small angle
        # Thus we aim for a big cosine (maximize)! We minimize -cosine.
        losses = - self.cos_loss(learned_directions, target_directions)
        return losses


class L2RegressionDirectionGetter(AbstractRegressionDirectionGetter):
    """
    Regression model.

    Loss = Pairwise distance = p_root(sum(|x_i|^P))
    * If sequence: averaged on time steps and sequences.
    """
    def __init__(self, input_size: int, dropout: float = None,
                 normalize_targets: float = False,
                 normalize_outputs: float = False,
                 add_eos_label: bool = False):
        # L2 Distance loss supports compressed streamlines
        super().__init__(input_size, dropout,
                         key='l2-regression',
                         supports_compressed_streamlines=True,
                         loss_description="torch's pairwise distance",
                         normalize_targets=normalize_targets,
                         normalize_outputs=normalize_outputs,
                         add_eos_label=add_eos_label)

        # Loss will be applied on the last dimension, by default in
        # PairWiseDistance
        self.l2_loss = PairwiseDistance()

        # Range of the L2: depends on the scaling of directions.
        # sqrt(2) if directions are normalized.
        # Generally, step size is smaller than one voxel, so smaller than
        # sqrt(2).
        self.scaling_factor_eos_loss = np.sqrt(2)

    def _compute_loss(self, learned_directions: Tensor,
                      target_directions: Tensor):
        losses = self.l2_loss(learned_directions, target_directions)
        return losses


class CosPlusL2RegressionDirectionGetter(AbstractRegressionDirectionGetter):
    def __init__(self, input_size: int, dropout: float = None,
                 normalize_targets: float = False,
                 normalize_outputs: float = False,
                 add_eos_label: bool = False):
        super().__init__(input_size, dropout,
                         key='cos-plus-l2-regression',
                         supports_compressed_streamlines=False,
                         loss_description='l2 + negative cosine similarity',
                         normalize_targets=normalize_targets,
                         normalize_outputs=normalize_outputs,
                         add_eos_label=add_eos_label)

        # Loss will be applied on the last dimension.
        self.cos_loss = CosineSimilarity(dim=-1)
        self.l2_loss = PairwiseDistance()

        self.scaling_factor_eos_loss = np.sqrt(2) + 2

    def _compute_loss(self, learned_directions: Tensor,
                      target_directions: Tensor):
        l2_losses = self.l2_loss(learned_directions, target_directions)
        cos_losses = -self.cos_loss(learned_directions, target_directions)
        losses = l2_losses + cos_losses

        return losses


class SphereClassificationDirectionGetter(AbstractDirectionGetterModel):
    """
    Classification model.

    Classes: Points on the sphere.

    Model: We use fully-connected (linear) network converting the outputs
    (at every step) to a 3D vector. Will use super to loop on layers, where
    layers are:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = 3

    Loss = negative log-likelihood.
    """
    def __init__(self, input_size: int, dropout: float = None,
                 sphere: str = 'symmetric724', add_eos_class=False,
                 smooth_labels: bool = False):
        """
        sphere: str
            An choice of dipy's Sphere.
        add_eos_class: bool
            If true, add an additional class for EOS.
        smooth_labels: bool
            If true, applies a Gaussian on the labels, as done in Deeptract.
        """
        super().__init__(input_size, dropout,
                         key='sphere-classification',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood')

        # Classes
        self.sphere_name = sphere
        sphere = dipy.data.get_sphere(sphere)
        self.torch_sphere = TorchSphere(sphere)
        self.smooth_labels = smooth_labels

        # EOS
        self.add_eos_class = add_eos_class

        nb_classes = sphere.vertices.shape[0]
        self.layers = init_2layer_fully_connected(input_size, nb_classes)

        # Loss will be defined in compute_loss, using torch distribution

    def move_to(self, device):
        super().move_to(device)
        self.torch_sphere.move_to(device)

    @property
    def params(self):
        params = super().params

        params.update({
            'sphere': self.sphere_name,
            'add_eos_class': self.add_eos_class,
            'smooth_labels': self.smooth_labels
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        logits = self.loop_on_layers(inputs, self.layers)
        return logits

    def prepare_targets(self, target_streamlines):
        # Find the closest class for each target direction
        target_dirs = compute_directions(target_streamlines)

        target_idx = convert_dirs_to_class(
            target_dirs, self.torch_sphere, smooth_label=self.smooth_labels,
            add_sos=False, add_eos=self.add_eos_class, device=self.device)
        return target_idx

    def compute_loss(self, logits_per_class: Tensor,
                     target_idx: Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's logits.
        """
        nb_class = self.torch_sphere.vertices.shape[0]
        if self.add_eos_class:
            nb_class += 1
        assert target_idx.shape[-1] == nb_class, \
            "Expecting targets reshaped as one-hot vectors for {} classes, " \
            "but got size {}. Use prepare_targets." \
            .format(nb_class, target_idx.shape[-1])

        # Create an official probability distribution from the logits
        # Applies a softmax
        distribution = Categorical(logits=logits_per_class)

        # Compute loss between distribution and target vertex
        nll_losses = -distribution.log_prob(target_idx)

        return _mean_and_weight(nll_losses)

    def _sample_tracking_direction_prob(self, logits_per_class: Tensor):
        """
        Sample a tracking direction on the sphere from the predicted class
        logits (=probabilities). If class = EOS, direction is [NaN, NaN, NaN].
        """
        # Sample a direction on the sphere
        sampler = Categorical(logits=logits_per_class)
        idx = sampler.sample()

        return self.torch_sphere.vertices[idx]

    def _get_tracking_direction_det(self, logits_per_class: Tensor):
        """
        Get the predicted class with highest logits (=probabilities).
        If class = EOS, direction is [NaN, NaN, NaN].
        """
        idx = logits_per_class.argmax(dim=1)

        return self.torch_sphere.vertices[idx]


class SingleGaussianDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model. The output is not a x,y,z value but the learned
    parameters of the gaussian representing the local direction.

    Not to be counfounded with the 3D Gaussian representing a tensor (means
    would be 0,0,0, the origin). This Gaussian's means represent the most
    probable direction, and variances represent the incertainty.

    Model: 2-layer NN for the means + 2-layer NN for the variances.

    Loss: Negative log-likelihood.
    """
    def __init__(self, input_size: int, dropout: float = None):
        # 3D gaussian supports compressed streamlines
        super().__init__(input_size, dropout,
                         key='gaussian',
                         supports_compressed_streamlines=True,
                         loss_description='negative log-likelihood')

        # Layers
        self.layers_mean = init_2layer_fully_connected(input_size, 3)
        self.layers_sigmas = init_2layer_fully_connected(input_size, 3)

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigma)
        sigmas = torch.exp(log_sigmas)

        return means, sigmas

    def compute_loss(self, learned_gaussian_params: Tuple[Tensor, Tensor],
                     target_dirs: Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.

        See the doc for explanation on the formulas:
        https://dwi-ml.readthedocs.io/en/latest/formulas.html
        """
        means, sigmas = learned_gaussian_params

        # Create an official function-probability distribution from the means
        # and variances
        distribution = MultivariateNormal(
            means, covariance_matrix=torch.diag_embed(sigmas ** 2))

        # Compute the negative log-likelihood from the difference between the
        # distribution and each target.
        nll_losses = -distribution.log_prob(target_dirs)

        return _mean_and_weight(nll_losses)

    def _sample_tracking_direction_prob(
            self, learned_gaussian_params: Tuple[Tensor, Tensor]):
        """
        From the gaussian parameters, sample a direction.
        """
        means, sigmas = learned_gaussian_params

        # Sample a final function in the chosen Gaussian
        # One direction per time step per sequence
        distribution = MultivariateNormal(
            means, covariance_matrix=torch.diag_embed(sigmas ** 2))
        direction = distribution.sample()

        return direction

    def _get_tracking_direction_det(self, learned_gaussian_params: Tensor):
        """
        Get the predicted class with highest logits (=probabilities).
        """
        # Returns the direction of the max of the Gaussian = the mean.
        means, sigmas = learned_gaussian_params

        return means


class GaussianMixtureDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model. The output is not a x,y,z value but the learned
    parameters of the distribution representing the local direction.

    Same as SingleGaussian but with more than one Gaussian. This should account
    for branching bundles, distributing probability across space at branching
    points.

    Model: (a 2-layer NN for the mean and a 2-layer NN for the sigma.) for
    each of N Gaussians.
    (Parameters:     N Gaussians * (1 mixture param + 3 means + 3 sigmas))

    Loss: Negative log-likelihood.
    """
    def __init__(self, input_size: int, dropout: float = None,
                 nb_gaussians: int = 3):
        # 3D Gaussian mixture supports compressed streamlines
        super().__init__(input_size, dropout,
                         key='gaussian-mixture',
                         supports_compressed_streamlines=True,
                         loss_description='negative log-likelihood')

        self.nb_gaussians = nb_gaussians

        # N Gaussians * (1 mixture param + 3 means + 3 sigmas)
        # (no correlations between each Gaussian)
        # Use separate *heads* for the mu and sigma, but concatenate gaussians

        self.layers_mixture = init_2layer_fully_connected(
            input_size, 1 * self.nb_gaussians)
        self.layers_mean = init_2layer_fully_connected(
            input_size, 3 * self.nb_gaussians)
        self.layers_sigmas = init_2layer_fully_connected(
            input_size, 3 * self.nb_gaussians)

        # Loss will be defined in compute_loss, using torch distribution

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

    def compute_loss(
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor],
            target_dirs):
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

        return _mean_and_weight(nll_losses)

    def _sample_tracking_direction_prob(
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor]):
        """
        From the gaussian mixture parameters, sample one of the gaussians
        using the mixture probabilities, then sample a direction from the
        selected gaussian.
        """
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
            self, learned_gaussian_params: Tuple[Tensor, Tensor, Tensor]):
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
        mixture_logits = gaussian_params[0].squeeze(dim=1)
        means = gaussian_params[1].reshape((-1, self.nb_gaussians, 3))
        sigmas = gaussian_params[2].reshape((-1, self.nb_gaussians, 3))

        return mixture_logits, means, sigmas


class FisherVonMisesDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model. The output is not a x,y,z value but the learned
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
    def __init__(self, input_size: int, dropout: float = None):
        super().__init__(input_size, dropout,
                         key='fisher-von-mises',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood')

        self.layers_mean = init_2layer_fully_connected(input_size, 3)
        self.layers_kappa = init_2layer_fully_connected(input_size, 1)

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the inputs through the fully-connected layer.

        Returns
        -------
        means : torch.Tensor with shape [batch_size x 3]
            ?
        kappas : torch.Tensor with shape [batch_size x 1]
            ?
        """
        means = self.loop_on_layers(inputs, self.layers_mean)
        # mean should be a unit vector for Fisher Von-Mises distribution
        means = torch.nn.functional.normalize(means, dim=-1)

        # Need to restrict kappa to a certain range, e.g. [0, 20]
        unbound_kappa = self.loop_on_layers(inputs, self.layers_kappa)
        kappas = torch.sigmoid(unbound_kappa) * 20

        # Squeeze the trailing dim, the kappa parameter is a scalar
        kappas = kappas.squeeze(dim=-1)

        return means, kappas

    def compute_loss(self, learned_fisher_params: Tuple[Tensor, Tensor],
                     target_dirs):
        """Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.

        See the doc for explanation on the formulas:
        https://dwi-ml.readthedocs.io/en/latest/formulas.html
        """
        # mu.shape : [flattened_sequences, 3]
        mu, kappa = learned_fisher_params

        log_prob = fisher_von_mises_log_prob(mu, kappa, target_dirs, eps)
        nll_losses = -log_prob

        return _mean_and_weight(nll_losses)

    def _sample_tracking_direction_prob(
            self, learned_fisher_params: Tuple[Tensor, Tensor]):
        """Sample directions from a fisher von mises distribution.
        """
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

    def _get_tracking_direction_det(self, learned_fisher_params: Tensor):
        # toDo. ?
        raise NotImplementedError

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


class FisherVonMisesMixtureDirectionGetter(AbstractDirectionGetterModel):
    """
    Compared to the version with 1 disbstribution, we now have an additional
    weight parameter alpha.
    """
    def __init__(self, input_size: int, dropout: float = None,
                 n_cluster: int = 3):
        super().__init__(input_size, dropout,
                         key='fisher-von-mises-mixture',
                         supports_compressed_streamlines=False,
                         loss_description='negative log-likelihood')

        self.n_cluster = n_cluster
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

    def compute_loss(self, outputs: Tuple[Tensor, Tensor],
                     target_dirs: Tensor):
        raise NotImplementedError

    def _sample_tracking_direction_prob(self, outputs: Tuple[Tensor, Tensor]):
        raise NotImplementedError

    def _get_tracking_direction_det(self, learned_fisher_params: Tensor):
        raise NotImplementedError


keys_to_direction_getters = \
    {'cosine-regression': CosineRegressionDirectionGetter,
     'l2-regression': L2RegressionDirectionGetter,
     'cosine-plus-l2-regression': CosPlusL2RegressionDirectionGetter,
     'sphere-classification': SphereClassificationDirectionGetter,
     'gaussian': SingleGaussianDirectionGetter,
     'gaussian-mixture': GaussianMixtureDirectionGetter,
     'fisher-von-mises': FisherVonMisesDirectionGetter,
     'fisher-von-mises-mixture': FisherVonMisesMixtureDirectionGetter}
