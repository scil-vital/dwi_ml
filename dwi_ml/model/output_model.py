from math import ceil
from typing import Any, List, Tuple

import dipy.data
import numpy as np
import torch
from torch.nn import (Linear, Dropout, ReLU, CosineSimilarity)
from torch.distributions import Categorical, MultivariateNormal
from torch.nn.modules.distance import PairwiseDistance

DESCRIPTION = """
MODELS:
    REGRESSION models: Simple regression to learn directly a direction (each 
    output size = 3 = [x,y,z]). This means that it is a deterministic output.
        - CosineRegressionOutput: Model: a 2-layer NN
                                  Loss: Cosine similarity
        - L2RegressionOutput: Model: 2 Linear layers
                              Loss: Pairwise distance
                              
    CLASSIFICATION models: That means that the output is a probability over all
    classes. We can decide how to sample the final direction. The classes depend 
    on the model.
        - SphereClassificationOutput: Model: a 2-layer NN *                                       
                                      Classes: 100 discrete points on the sphere
                                      (dipy.data.get_sphere('symmetric724'))
                                      Loss: Negative log-likelyhood

    GAUSSIAN models: Contrary to regression, who would learn the parameters 
    (hidden through the weights) that would represent a function h such that
    y ~ h(x)  (we learn the *parameters*), gaussian processes learn directly 
    the *function probability*. See for ex here
    https://blog.dominodatalab.com/fitting-gaussian-process-models-python/
    The model learns to represent the mean and variance of all the functions 
    that could represent the data. 
        - SingleGaussianOutput: Model: a 2-layer NN for the mean and a 2-layer 
                                NN for the variance.*
                                Loss: Negative log-likelihood
        - GaussianMixtureOutput: Model: (a 2-layer NN for the mean and a 2-layer 
                                 NN for the variance.) for each of N Gaussians.*                  
                                 Loss: Negative log-likelihood.

    FISHER VON MISES models: This model provides probabilistic outputs using the
    Fisher - von Mises distribution, which resembles a gaussian on the sphere. 
    As such, it does not require unit normalization when sampling, and should be
    more stable while training.                                                                                          # toDo. Can we have a bit more (easy) explanation?
 
* p.s. Torch kind of does a softmax after although it is not explicit.
                                                                         
INPUTS:  Def: Here, we call 'input' the output of your experiment model. Ex, RNN.
         Type: torch.tensor
         Size:
             - Sequence models: [batch_size*seq_len, nb_features]
              where seq_len is the nb of time steps in the sequence.
              So the input sequences are concatenated.
             - Local models: [batch_size, nb_features]

OUTPUTS: Def: The model final outputs.
         Type: torch.tensor
         Size: 
             - Sequence model: [batch_size*seq_len, 3]
             - Local models: [batch_size, 3]
         
TARGETS: Def: The target values (real Y) for the batch
         Type: Could be `PackedSequence.data`
         Size: 
             - Sequence models: [batch_size*seq_len, 3]
             - Local models: [batch_size, 3]
"""


class BaseTrackingOutputModel(torch.nn.Module):
    """
    Default static class attribute, to be redefined by sub-classes.

    Prepares the main functions. All models will be similar in the way that they
    all define layers. Then, we always apply self.loop_on_layers()

    input  -->  layer_i --> ReLu --> dropout -->  last_layer --> output
                  |                     |
                  -----------------------
    """
    supportsCompressedStreamlines = False

    def __init__(self, dropout: float = None):
        """ Prepares the dropout and ReLU sub-layers"""
        super().__init__()
        self.dropout = dropout
        if self.dropout:
            self.dropout_sublayer = Dropout(self.dropout)
        else:
            self.dropout_sublayer = lambda x: x
        self.relu_sublayer = ReLU()

    def loop_on_layers(self, inputs: torch.tensor,
                       layers: List[torch.nn.Module]):
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

    def forward(self, inputs: torch.Tensor):
        # Will be implemented by each class
        raise NotImplementedError

    def compute_loss(self, outputs: Any, targets: torch.Tensor) \
            -> torch.Tensor:
        # Will be implemented by each class
        raise NotImplementedError

    def sample_tracking_directions(self, outputs: torch.Tensor) \
            -> torch.Tensor:
        # Will be implemented by each class
        raise NotImplementedError


class CosineRegressionOutput(BaseTrackingOutputModel):
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

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(dropout)

        # Layers
        hidden_size = ceil(input_size / 2)
        h1 = Linear(input_size, hidden_size)
        h2 = Linear(hidden_size, 3)
        self.layers = [h1, h2]

        # Loss will be applied on the last dimension.
        self.loss = CosineSimilarity(dim=-1)

    def forward(self, inputs: torch.Tensor):
        """ Run the inputs through the loop on layers.  """
        output = self.loop_on_layers(inputs, self.layers)
        return output

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
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

    def sample_tracking_directions(self, outputs: torch.Tensor) \
            -> torch.Tensor:
        """
        In this case, the output is directly a direction, so we can use it as
        is for the tracking.
        """
        return outputs


class L2RegressionOutput(BaseTrackingOutputModel):
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
        super().__init__(dropout)

        # Layers
        hidden_size = ceil(input_size / 2)
        h1 = Linear(input_size, hidden_size)
        h2 = Linear(hidden_size, 3)
        self.layers = [h1, h2]

        # Loss will be applied on the last dimension, by default in
        # PairWiseDistance
        self.loss = PairwiseDistance()

    def forward(self, inputs: torch.Tensor):
        """
        Run the inputs through the loop on layers.
        """
        output = self.loop_on_layers(inputs, self.layers)
        return output

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        """Compute the average negative cosine similarity between the computed
        directions and the target directions.
        """
        # If outputs and targets are of shape (N, D), losses is of shape N
        losses = self.loss(outputs, targets)
        mean_loss = losses.mean()
        return mean_loss

    def sample_tracking_directions(self, outputs: torch.Tensor)\
            -> torch.Tensor:
        """
        In this case, the output is directly a direction, so we can use it as
        is for the tracking.
        """
        return outputs


class SphereClassificationOutput(BaseTrackingOutputModel):
    """
    Classification model.

    Classes: 100 points on the sphere (dipy.data.get_sphere('symmetric724'))

    Model: Same as before, a 2-layer NN.

    Loss = negative log-likelihood.
    """

    def __init__(self, input_size: int, dropout: float = None):
        # Prepare the dropout, Relu, loop:
        super().__init__(dropout)

        # Classes
        self.sphere = dipy.data.get_sphere('symmetric724')
        self.vertices = torch.as_tensor(self.sphere.vertices,
                                        dtype=torch.float32)
        output_size = self.sphere.vertices.shape[0]

        # Layers
        hidden_size = ceil(input_size / 2)
        h1 = Linear(input_size, hidden_size)
        h2 = Linear(hidden_size, output_size)
        self.layers = [h1, h2]

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: torch.Tensor):
        """
        Run the inputs through the loop on layers.
        """
        logits = self.loop_on_layers(inputs, self.layers)
        return logits

    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's logits.
        """

        # Find the closest idx per time step per sequence in the batch:
        target_idx = self._find_closest_vertex(targets)
        target_idx_tensor = torch.as_tensor(target_idx, dtype=torch.int16,
                                            device=outputs.device)

        # Create an official probability distribution from the logits
        distribution = Categorical(logits=outputs)

        # Compute loss between distribution and target vertex
        nll_losses = -distribution.log_prob(target_idx_tensor)

        # Average on timesteps and sequences in batch
        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(self, outputs: torch.Tensor) \
            -> torch.Tensor:
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


class SingleGaussianOutput(BaseTrackingOutputModel):
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
        super().__init__(dropout)

        # Layers
        hidden_size = ceil(input_size / 2)

        h1_mean = Linear(input_size, hidden_size)
        h2_mean = Linear(hidden_size, 3)
        self.layers_mean = [h1_mean, h2_mean]

        h1_variance = Linear(input_size, hidden_size)
        h2_variance = Linear(hidden_size, 3)
        self.layers_variance = [h1_variance, h2_variance]

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: torch.Tensor):
        """
        Run the inputs through the loop on layers.
        """
        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigma)
        variances = torch.exp(log_sigmas)

        return means, variances

    def compute_loss(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                     targets: torch.Tensor):
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

    def sample_tracking_directions(self,
                                   outputs: Tuple[torch.Tensor, torch.Tensor]) \
            -> torch.Tensor:
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


class GaussianMixtureOutput(BaseTrackingOutputModel):
    """
    Same as SingleGaussian but with more than one Gaussian. This should account
    for branching bundles, distributing probability across space at branching
    points.

    Model: (a 2-layer NN for the mean and a 2-layer NN for the variance.) for
    each of N Gaussians.
    (Parameters:     N Gaussians * (1 mixture param + 3 means + 3 variances))

    Loss: Negative log-likelihood.
    """
    # 3D Gaussian mixture supports compressed streamlines
    supportsCompressedStreamlines = True

    def __init__(self, input_size: int, dropout: float = None,
                 nb_gaussians: int = 3):
        # Prepare the dropout, Relu, loop:
        super().__init__(dropout)

        self.n_gaussians = 3

        # Layers
        hidden_size = ceil(input_size / 2)

        h1_mixture = Linear(input_size, hidden_size)
        h2_mixture = Linear(hidden_size, self.n_gaussians)
        self.layers_mixture = [h1_mixture, h2_mixture]

        # 3 means for each gaussian
        h1_mean = Linear(input_size, hidden_size)
        h2_mean = Linear(hidden_size, 3 * self.n_gaussians)
        self.layers_mean = [h1_mean, h2_mean]

        # 3 variances for each gaussian
        h1_variance = Linear(input_size, hidden_size)
        h2_variance = Linear(hidden_size, 3 * self.n_gaussians)
        self.layers_variance = [h1_variance, h2_variance]

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: torch.Tensor):
        """
        Run the inputs through the loop on layers.
        """
        mixture_logits = self.loop_on_layers(inputs, self.layers_mixture_logits)

        means = self.loop_on_layers(inputs, self.layers_mean)

        log_sigmas = self.loop_on_layers(inputs, self.layers_sigma)
        sigmas = torch.exp(log_sigmas)

        return mixture_logits, means, sigmas

    def compute_loss(self,
                     outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                     targets: torch.Tensor):
        """
        Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.
        """
        # Shape : [batch_size*seq_len, n_gaussians, 3] or
        # [batch_size, n_gaussians, 3}
        mixture_logits, means, variances = self._get_gaussian_parameters(outputs)

        # Take softmax for the mixture parameters
        mixture_probs = torch.softmax(mixture_logits, dim=-1)

        # Compute the difference between the means and the targets
        diff = means - targets[:, None, :]

        # Compute mahalanobis distance
        square_m_distance = (diff / variances).pow(2).sum(dim=-1)

        # Compute the log-probabilities and log-likelihood
        log_det = variances.log().sum(dim=-1)
        log_2pi = np.log(2 * np.pi).astype(np.float32)
        gaussians_log_prob = -0.5 * ((3 * log_2pi) + square_m_distance)\
                             - log_det
        nll_losses = -torch.logsumexp(mixture_probs.log() + gaussians_log_prob,
                                    dim=-1)
        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(
            self, outputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> torch.Tensor:
        """
        From the gaussian mixture parameters, sample one of the gaussians
        using the mixture probabilities, then sample a direction from the
        selected gaussian.
        """
        mixture_logits, means, variances = self._get_gaussian_parameters(outputs)

        # Create probability distribution and sample a gaussian per point
        # (or per time step per sequence)
        mixture_distribution = Categorical(logits=mixture_logits)
        mixture_id = mixture_distribution.sample()

        # For each point in the batch (of concatenate sequence) take the mean
        # and variance parameters. Note. Means and variances are of shape
        # [batch_size*seq_len, n_gaussians, 3] or [batch_size, n_gaussians, 3]
        component_means = means[:, mixture_id, :]
        component_sigmas = variances[:, mixture_id, :]

        # Sample a final function in the chosen Gaussian
        # One direction per timestep per sequence.
        component_distribution = MultivariateNormal(
            component_means,
            covariance_matrix=torch.diag_embed(component_sigmas ** 2))
        direction = component_distribution.sample()

        return direction

    def _get_gaussian_parameters(
            self,
            model_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """From the model output, extract the mixture parameters, the means
        and the variances, all reshaped according to the number of components
        and the dimensions (3D). i.e. [batch_size, n_gaussians] for the mixture
        logit and [batch_size, n_gaussians, 3] for the means and variances.
        """
        mixture_logits = model_output[0].squeeze(dim=1)
        means = model_output[1].reshape((-1, self.n_gaussians, 3))
        variances = model_output[2].reshape((-1, self.n_gaussians, 3))

        return mixture_logits, means, variances


class FisherVonMisesOutput(BaseTrackingOutputModel):
    """
    This model provides probabilistic outputs using the Fisher - von Mises
    distribution [1][2], which resembles a gaussian on the sphere. As such,
    it does not require unit normalization when sampling, and should be more
    stable while training.

    We sample using rejection sampling defined in [3], implemented in [4].

    Ref:
    [1]: https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    [2]: http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
    [3]: Directional Statistics (Mardia and Jupp, 1999)
    [4]: https://github.com/jasonlaska/spherecluster
    """

    def __init__(self, input_size: int, dropout: float = None,
                 nb_gaussians: int = 3):
        # Prepare the dropout, Relu, loop:
        super().__init__(dropout)

        # Layers
        hidden_size = ceil(input_size / 2)

        # Mean (3D)
        h1_mean = Linear(input_size, hidden_size)
        h2_mean = Linear(hidden_size, 3)
        self.layers_mean = [h1_mean, h2_mean]

        # Kappa (1 value)
        h1_kappa = Linear(input_size, hidden_size)
        h2_kappa = Linear(hidden_size, 1)
        self.layers_variance = [h1_kappa, h2_kappa]

        # Loss will be defined in compute_loss, using torch distribution

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

    def compute_loss(self, outputs: Tuple[torch.Tensor, torch.Tensor],
                     targets: torch.Tensor):
        """Compute the negative log-likelihood for the targets using the
        model's mixture of gaussians.
        """
        # mu.shape : [flattened_sequences, 3]
        mu, kappa = outputs

        log_2pi = np.log(2 * np.pi).astype(np.float32)

        # Add an epsilon in case kappa is too small (i.e. a uniform
        # distribution)
        eps = 1e-6
        log_diff_exp_kappa = torch.log(torch.exp(kappa)
                                       - torch.exp(-kappa)
                                       + eps)

        batch_dot_product = torch.sum(mu * targets, dim=1)

        nll_losses = -torch.log(kappa) + log_2pi + log_diff_exp_kappa - \
                   (kappa * batch_dot_product)

        mean_loss = nll_losses.mean()

        return mean_loss

    def sample_tracking_directions(
            self, outputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
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


OUTPUT_KEY_TO_CLASS = {'cosine-regression': CosineRegressionOutput,
                       'l2-regression': L2RegressionOutput,
                       'sphere-classification': SphereClassificationOutput,
                       'gaussian': SingleGaussianOutput,
                       'gaussian-mixture': GaussianMixtureOutput,
                       'fisher-von-mises': FisherVonMisesOutput}
