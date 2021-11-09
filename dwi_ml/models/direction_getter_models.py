# -*- coding: utf-8 -*-
from math import ceil
from typing import Any, List, Tuple

import dipy.data
import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical, MultivariateNormal
from torch.nn import (CosineSimilarity, Dropout, Linear, ModuleList, ReLU)
from torch.nn.modules.distance import PairwiseDistance

from dwi_ml.models.main_models import ModelAbstract
from dwi_ml.models.utils_for_gaussians import independent_gaussian_log_prob
from dwi_ml.models.utils_for_fisher_von_mises import fisher_von_mises_log_prob

DESCRIPTION = """
MODELS:
    REGRESSION models: Simple regression to learn directly a direction (each
    output size = 3 = [x,y,z]). This means that it is a deterministic output.
        - CosineRegressionDirectionGetter:
                Model: a 2-layer NN
                Loss: Cosine similarity between the computed direction and the
                      provided target direction.
                      = cos(theta) = (y*t)/(||y|| ||t||)
                      The mean of loss values for each step of the streamline
                      is computed.
        - L2RegressionDirectionGetter:
                Model: 2 Linear layers
                Loss: Pairwise distance
                      = sqrt(sum(t_i - y_i)^2)
                      The mean of loss values for each step of the streamline
                      is computed.

    CLASSIFICATION models: That means that the output is a probability over all
    classes. We can decide how to sample the final direction. The classes
    depend on the model.
        - SphereClassificationDirectionGetter:
               Model: a 2-layer NN
               Classes: 724 discrete points on the sphere
                        (dipy.data.get_sphere('symmetric724'))
               Loss: Negative log-likelyhood from a softmax (integrated inside
               torch) = cross-entropy

    GAUSSIAN models: Contrary to regression, which would learn the parameters
    (hidden through the weights) that would represent a function h such that
    y ~ h(x)  (we learn the *parameters*), gaussian processes learn directly
    the *function probability*. See for ex here
    https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
    The model learns to represent the mean and variance of all the functions
    that could represent the data. We can decide how to sample the final
    direction.
        - SingleGaussianDirectionGetter:
                Model: a 2-layer NN for the mean and a 2-layer NN for the
                       variance.
                Loss: Negative log-likelihood
        - GaussianMixtureDirectionGetter:
                Model: a 2-layer NN for the mean and a 2-layer NN for the
                       variance, for each of N Gaussians.
                Loss: Negative log-likelihood

    FISHER VON MISES models: This model provides probabilistic outputs using
    the Fisher - von Mises distribution, which resembles a gaussian on the
    sphere. As such, it does not require unit normalization when sampling, and
    should be more stable while training. We can decide how to sample the final
    direction.
                Model:  a 2-layer NN for the mean and a 2-layer NN for the
                       'kappas'.
                Loss: Negative log-likelihood

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
         Type: Could be `PackedSequence.data`
         Size: 
             - Sequence models: [batch_size*seq_len, 3]
             - Local models: [batch_size, 3]
"""
CHOSEN_SPHERE = 'symmetric724'
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


class AbstractDirectionGetterModel(ModelAbstract):
    """
    Default static class attribute, to be redefined by sub-classes.

    Prepares the main functions. All models will be similar in the way that
    they all define layers. Then, we always apply self.loop_on_layers()

    input  -->  layer_i --> ReLu --> dropout -->  last_layer --> output
                  |                     |
                  -----------------------
    """
    supportsCompressedStreamlines = False

    def __init__(self, input_size, dropout: float = 0.):
        """ Prepares the dropout and ReLU sub-layers"""
        super().__init__()

        self.input_size = input_size

        if dropout and (dropout < 0 or dropout > 1):
            raise ValueError('Dropout rate should be between 0 and 1.')
        self.dropout = dropout
        if self.dropout:
            self.dropout_sublayer = Dropout(self.dropout)
        else:
            self.dropout_sublayer = lambda x: x

        self.relu_sublayer = ReLU()

    @property
    def params(self):
        params = {
            'dropout': self.dropout,
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
        # Will be implemented by each class
        raise NotImplementedError

    def compute_loss(self, outputs: Any, targets: Tensor) -> Tensor:
        # Will be implemented by each class
        raise NotImplementedError

    def sample_tracking_directions(self, outputs: Tensor) -> Tensor:
        # Will be implemented by each class
        # For deterministic models, this is straightforward.
        # For probabilistic models, we need to sample the tracking directions.
        raise NotImplementedError


class CosineRegressionDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model.

    Here we use fully-connected (linear) layers converting the outputs
    (at every step) to a 3D vector. Will use super to loop on layers, where
    layers are:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = 3

    Loss = negative cosine similarity.
    * If sequence: averaged on time steps and sequences.
    """
    # Compressed streamlines not supported
    supportsCompressedStreamlines = False

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(input_size, dropout)

        self.layers = init_2layer_fully_connected(input_size, 3)

        # Loss will be applied on the last dimension.
        self.loss = CosineSimilarity(dim=-1)

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'cosine-regression',
            'loss': 'negative cosine similarity'
        })
        return params

    def forward(self, inputs: Tensor):
        """ Run the inputs through the loop on layers.  """
        output = self.loop_on_layers(inputs, self.layers)
        return output

    def compute_loss(self, outputs: Tensor, targets: Tensor):
        """
        Compute the average negative cosine similarity between the computed
        directions and the target directions.

        Returns
        -------
        mean_loss : torch.Tensor
            The loss between the outputs and the targets, averaged across
            timesteps (and sequences).
        """
        # loss has shape [n_sequences*seq_len]
        losses = -self.loss(outputs, targets)
        mean_loss = losses.mean()
        return mean_loss

    def sample_tracking_directions(self, outputs: Tensor) -> Tensor:
        """
        In this case, the output is directly a direction, so we can use it as
        is for the tracking.
        """
        return outputs


class L2RegressionDirectionGetter(AbstractDirectionGetterModel):
    """
    Regression model.

    Same as in CosineRegressionOutput, we use a 2-layer NN:
        1. Linear1:  output size = ceil(input_size/2)
        2. Linear2:  output size = 3

    Loss = Pairwise distance = p_root(sum(|x_i|^P))
    * If sequence: averaged on time steps and sequences.
    """
    # L2 Distance loss supports compressed streamlines
    supportsCompressedStreamlines = True

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(input_size, dropout)

        self.layers = init_2layer_fully_connected(input_size, 3)

        # Loss will be applied on the last dimension, by default in
        # PairWiseDistance
        self.loss = PairwiseDistance()

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'l2-regression',
            'loss': 'pairwise distance'
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        output = self.loop_on_layers(inputs, self.layers)
        return output

    def compute_loss(self, outputs: Tensor, targets: Tensor):
        """Compute the average negative cosine similarity between the computed
        directions and the target directions.
        """
        # If outputs and targets are of shape (N, D), losses is of shape N
        losses = self.loss(outputs, targets)
        mean_loss = losses.mean()
        return mean_loss

    def sample_tracking_directions(self, outputs: Tensor) -> Tensor:
        """
        In this case, the output is directly a direction, so we can use it as
        is for the tracking.
        """
        return outputs


class SphereClassificationDirectionGetter(AbstractDirectionGetterModel):
    """
    Classification model.

    Classes: 724 points on the sphere (dipy.data.get_sphere('symmetric724'))

    Model: Same as before, a 2-layer NN.

    Loss = negative log-likelihood.
    """
    supportsCompressedStreamlines = False

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(input_size, dropout)

        # Classes
        self.sphere = dipy.data.get_sphere(CHOSEN_SPHERE)
        self.vertices = torch.as_tensor(self.sphere.vertices,
                                        dtype=torch.float32)
        output_size = self.sphere.vertices.shape[0]

        self.layers = init_2layer_fully_connected(input_size, output_size)

        # Loss will be defined in compute_loss, using torch distribution

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'sphere-classification',
            'loss': 'negative log-likelihood'
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        logits = self.loop_on_layers(inputs, self.layers)
        return logits

    def compute_loss(self, outputs: Tensor, targets: Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's logits.
        """

        # Find the closest class for each target direction
        target_idx = self._find_closest_vertex(targets)
        target_idx_tensor = torch.as_tensor(target_idx, dtype=torch.int16,
                                            device=outputs.device)

        # Create an official probability distribution from the logits
        # Applies a softmax
        distribution = Categorical(logits=outputs)

        # Compute loss between distribution and target vertex
        nll_losses = -distribution.log_prob(target_idx_tensor)

        # Average on timesteps and sequences in batch
        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(self, outputs: Tensor) -> Tensor:
        """
        Sample a tracking direction on the sphere from the predicted class
        logits (=probabilities).
        """

        # Sample a direction on the sphere
        sampler = Categorical(logits=outputs)
        idx = sampler.sample()

        # One direction per time step per sequence
        direction = self.vertices[idx]

        return direction

    def _find_closest_vertex(self, directions):
        """
        Returns vertices by index of cosine distance to a direction vector
        """
        # First send the vertices on the right device, i.e. same as target
        # directions
        if self.vertices.device != directions.device:
            self.vertices = self.vertices.to(device=directions.device)

        # We will use cosine similarity to find nearest vertex
        cosine_similarity = torch.matmul(directions, self.vertices.t())

        # Ordering by similarity. On the last dimension = per time step per
        # sequence in the batch.
        index = torch.argmax(cosine_similarity, dim=-1).type(torch.int16)

        return index


class SingleGaussianDirectionGetter(AbstractDirectionGetterModel):
    """
    Classification model

    Classes: 3D multivariate gaussian.

    Model: 2-layer NN for the mean 2-layer NN for the variance

    Loss: Negative log-likelihood.
    """
    # 3D gaussian supports compressed streamlines
    supportsCompressedStreamlines = True

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(input_size, dropout)

        # Layers
        self.layers_mean = init_2layer_fully_connected(input_size, 3)
        self.layers_variance = init_2layer_fully_connected(input_size, 3)

        # Loss will be defined in compute_loss, using torch distribution

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'gaussian',
            'loss': 'negative log-likelihood'
        })
        return params

    def forward(self, inputs: Tensor):
        """
        Run the inputs through the loop on layers.
        """
        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigma)
        variances = torch.exp(log_sigmas)

        return means, variances

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], targets: Tensor):
        """Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.
        """
        means, variances = outputs

        # Create an official function-probability distribution from the means
        # and variances
        distribution = MultivariateNormal(
            means, covariance_matrix=torch.diag_embed(variances ** 2))

        # Compute the negative log-likelihood from the difference between the
        # distribution and each target.
        nll_losses = -distribution.log_prob(targets)
        mean_loss = nll_losses.mean()
        return mean_loss

    def sample_tracking_directions(self, outputs: Tuple[Tensor, Tensor]) \
            -> Tensor:
        """
        From the gaussian parameters, sample a direction.
        """
        means, variances = outputs

        # Sample a final function in the chosen Gaussian
        # One direction per time step per sequence
        distribution = MultivariateNormal(
            means, covariance_matrix=torch.diag_embed(variances ** 2))
        direction = distribution.sample()

        return direction


class GaussianMixtureDirectionGetter(AbstractDirectionGetterModel):
    """
    Same as SingleGaussian but with more than one Gaussian. This should account
    for branching bundles, distributing probability across space at branching
    points.

    Model: (a 2-layer NN for the mean and a 2-layer NN for the sigma.) for
    each of N Gaussians.
    (Parameters:     N Gaussians * (1 mixture param + 3 means + 3 sigmas))

    Loss: Negative log-likelihood. Tyically, in the literature, Gaussian
    mixtures are used with expectation-maximisation (EM). Here we simply update
    the mixture parameters and the Gaussian parameters jointly, similar to
    GMM in [1].

    Refs:
    [1] https://github.com/jych/cle/blob/master/cle/cost/__init__.py
    """
    # 3D Gaussian mixture supports compressed streamlines
    supportsCompressedStreamlines = True

    def __init__(self, input_size: int, dropout: float = None,
                 nb_gaussians: int = 3):
        # Prepare the dropout, Relu, loop:
        super().__init__(dropout)

        self.n_gaussians = nb_gaussians

        # N Gaussians * (1 mixture param + 3 means + 3 sigmas)
        # (no correlations between each Gaussian)
        # Use separate *heads* for the mu and sigma, but concatenate gaussians

        self.layers_mixture = init_2layer_fully_connected(
            input_size, 1 * self.n_gaussians)
        self.layers_mean = init_2layer_fully_connected(
            input_size, 3 * self.n_gaussians)
        self.layers_sigmas = init_2layer_fully_connected(
            input_size, 3 * self.n_gaussians)

        # Loss will be defined in compute_loss, using torch distribution

    @property
    def params(self):
        params = super().params
        params.update({
            'key': 'gaussian-mixture',
            'nb_gaussians': self.n_gaussians,
            'loss': 'negative log-likelihood'
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

    def compute_loss(self, outputs: Tuple[Tensor, Tensor, Tensor],
                     targets: Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.
        """
        # Note. Shape of targets: [batch_size*seq_len, 3] or [batch_size, 3]

        # Shape : [batch_size*seq_len, n_gaussians, 3] or
        #         [batch_size, n_gaussians, 3]
        mixture_logits, means, sigmas = self._get_gaussian_parameters(outputs)

        # Take softmax for the mixture parameters
        mixture_probs = torch.softmax(mixture_logits, dim=-1)

        gaussians_log_prob = independent_gaussian_log_prob(targets[:, None, :],
                                                           means, sigmas)

        nll_losses = -torch.logsumexp(mixture_probs.log() + gaussians_log_prob,
                                      dim=-1)
        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(self,
                                   outputs: Tuple[Tensor, Tensor, Tensor]) \
            -> Tensor:
        """
        From the gaussian mixture parameters, sample one of the gaussians
        using the mixture probabilities, then sample a direction from the
        selected gaussian.
        """
        mixture_logits, means, sigmas = self._get_gaussian_parameters(outputs)

        # Create probability distribution and sample a gaussian per point
        # (or per time step per sequence)
        mixture_distribution = Categorical(logits=mixture_logits)
        mixture_id = mixture_distribution.sample()

        # For each point in the batch (of concatenate sequence) take the mean
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

    def _get_gaussian_parameters(self,
                                 model_output: Tuple[Tensor, Tensor, Tensor]) \
            -> Tuple[Tensor, Tensor, Tensor]:
        """From the model output, extract the mixture parameters, the means
        and the sigmas, all reshaped according to the number of components
        and the dimensions (3D). i.e. [batch_size, n_gaussians] for the mixture
        logit and [batch_size, n_gaussians, 3] for the means and sigmas.
        """
        mixture_logits = model_output[0].squeeze(dim=1)
        means = model_output[1].reshape((-1, self.n_gaussians, 3))
        sigmas = model_output[2].reshape((-1, self.n_gaussians, 3))

        return mixture_logits, means, sigmas


class FisherVonMisesDirectionGetter(AbstractDirectionGetterModel):
    """
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
    supportsCompressedStreamlines = False

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(input_size, dropout)

        self.layers_mean = init_2layer_fully_connected(input_size, 3)
        self.layers_kappa = init_2layer_fully_connected(input_size, 1)

        # Loss will be defined in compute_loss, using torch distribution

    @property
    def params(self):
        params = super().params  # type: dict
        params.update({
            'key': 'fisher-von-mises',
            'loss': 'negative log-likelihood'
        })
        return params

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        """Run the inputs through the fully-connected layer.

        Returns
        -------
        means : torch.Tensor with shape [batch_size x 3]
            The gaussian components means
        kappas : torch.Tensor with shape [batch_size x 1]
            The gaussian components standard deviations
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

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], targets: Tensor):
        """Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.
        """
        # mu.shape : [flattened_sequences, 3]
        mu, kappa = outputs

        log_prob = fisher_von_mises_log_prob(mu, kappa, targets, eps)
        nll_losses = -log_prob

        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(self, outputs: Tuple[Tensor, Tensor]) \
            -> Tensor:
        """Sample directions from a fisher von mises distribution.
        """
        # mu.shape : [flattened_sequences, 3]
        mus, kappas = outputs

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

        directions = torch.as_tensor(result, dtype=torch.float32)

        return directions

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
        super().__init__(input_size, dropout)

        self.n_cluster = n_cluster
        raise NotImplementedError

    @property
    def params(self):
        params = super().params
        params.update({
            'key': 'fisher-von-mises-mixture',
            'n_cluster': self.n_cluster
        })
        return params

    def forward(self, inputs: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    def compute_loss(self, outputs: Tuple[Tensor, Tensor], targets: Tensor):
        raise NotImplementedError

    def sample_tracking_directions(self, outputs: Tuple[Tensor, Tensor]) \
            -> Tensor:
        raise NotImplementedError


keys_to_direction_getters = \
    {'cosine-regression': CosineRegressionDirectionGetter,
     'l2-regression': L2RegressionDirectionGetter,
     'sphere-classification': SphereClassificationDirectionGetter,
     'gaussian': SingleGaussianDirectionGetter,
     'gaussian-mixture': GaussianMixtureDirectionGetter,
     'fisher-von-mises': FisherVonMisesDirectionGetter,
     'fisher-von-mises-mixture': FisherVonMisesMixtureDirectionGetter}
