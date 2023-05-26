#!/usr/bin/env python
import logging
from typing import Union, Tuple

import numpy as np
from dipy.data import get_sphere
from scipy.spatial.distance import (cosine, euclidean, mahalanobis)
import torch
from torch import logsumexp
from torch.nn.utils.rnn import PackedSequence
from torch.nn.functional import softmax

from dwi_ml.models.direction_getter_models import (
    CosineRegressionDG, FisherVonMisesDG,
    GaussianMixtureDG, L2RegressionDG,
    SingleGaussianDG, SphereClassificationDG, AbstractDirectionGetterModel)
from dwi_ml.models.utils.fisher_von_mises import (
    fisher_von_mises_log_prob_vector)

"""
Included utils are:
    test_cosine_regression_loss()
        - identical vectors
        - vectors with same angles
        - vectors at 90 degrees
        - vectors at 180 degrees
        - comparison with scipy.spatial.distance.cosine
    test_l2regression_loss()
        - identical vectors
        - comparison with scipy.spatial.distance.euclidean
    test_gaussian_loss()
        - x = mu
        - comparison with (manual + scipy)
    test_mixture_loss()
        - comparison with (manual + scipy)
"""
# toDo
#  test fisher von mises

# Using float32, we have come accross differences of the order 1e-3 for
# Fisher-von-Mises, depending on the random values.
tol = 1e-3
d = 3

logging.getLogger().setLevel(level='DEBUG')


def _independent_gaussian_log_prob_vector(x, mus, sigmas):
    """
    Equivalent to the torch method in model.utils.gaussians. Easier to test.

    Parameters
    ----------
    x = the variable
    mu = mean of the gaussian (x,y,z directions)
    sigmas = standard deviation of the gaussian (x,y,z directions)
    """
    # The inverse of a diagonal matrix is just inverting values on the
    # diagonal
    cov_inv = torch.eye(d) * (1 / sigmas ** 2)

    # sum(log) = log(prod)
    logpdf = (-d / 2 * torch.log(2 * torch.as_tensor(torch.pi)) -
              torch.log(torch.prod(sigmas))) \
        - 0.5 * mahalanobis(x[:3], mus, cov_inv) ** 2
    return logpdf


def _get_random_vector(size=3):
    scaling = np.random.randint(1, 9)
    return (np.random.randn(size) * scaling).astype(np.float32)


def _prepare_packedsequence(a):
    if not isinstance(a, PackedSequence):
        a = PackedSequence(data=(torch.as_tensor(a[None, :],
                                                 dtype=torch.float32)),
                           batch_sizes=torch.as_tensor([1]))
        return a


def _verify_loss(streamline: Union[torch.Tensor, list],
                 fake_model_output: Union[np.ndarray, list, torch.Tensor,
                 Tuple], expected_loss: Union[torch.Tensor, np.ndarray, float],
                 model: AbstractDirectionGetterModel,
                 expected_eos_loss=torch.as_tensor(0.)):
    # Convert types
    if not isinstance(streamline, torch.Tensor):
        streamline = torch.as_tensor(np.asarray(streamline),
                                     dtype=torch.float32)
    if (isinstance(fake_model_output, np.ndarray) or
            isinstance(fake_model_output, list)):
        fake_model_output = torch.as_tensor(fake_model_output,
                                            dtype=torch.float32)
    if (isinstance(expected_loss, np.ndarray) or
            isinstance(expected_loss, float)):
        expected_loss = torch.as_tensor(expected_loss, dtype=torch.float32)

    # Verify 2D
    if isinstance(fake_model_output, torch.Tensor) and \
            len(fake_model_output.shape) < 2:
        fake_model_output = fake_model_output[None, :]

    # Compute loss and verify
    computed_loss, _ = model.compute_loss([fake_model_output], [streamline])

    assert np.allclose(computed_loss, expected_loss, atol=tol), \
        "Expected loss {} but got {}.\n" \
        "      - Streamline: {}    ( = dir {})\n" \
        "      - Fake model output: {}\n" \
        .format(expected_loss + expected_eos_loss, computed_loss,
                streamline, streamline[1, :] - streamline[0, :],
                fake_model_output)


def test_cosine_regression_loss():
    logging.debug('Testing cosine regression loss')

    model = CosineRegressionDG(input_size=3)
    streamline = [[0., 0, 0], [1., 0, 0]]

    logging.debug("  - Identical vectors x: expecting -1 (+1)")
    good_dir = [1., 0, 0]
    expected_loss = 0.
    _verify_loss(streamline, good_dir, expected_loss, model)

    logging.debug("  - Vectors with same angle: expecting -1 (+1)")
    scales = np.random.random(20) * 20
    for s in scales:
        good_dir_bad_scale = [s, 0., 0]
        expected_loss = 0.
        _verify_loss(streamline, good_dir_bad_scale, expected_loss, model)

    logging.debug("  - Vectors with at 90 degrees 1: expecting 0 (+1)")
    dir_90 = [0., 1, 0]
    expected_loss = 1.
    _verify_loss(streamline, dir_90, expected_loss, model)

    logging.debug("  - Vectors with at 180 degrees: expecting 1 (+1)")
    dir_90 = [-1., 0, 0]
    expected_loss = 2.
    _verify_loss(streamline, dir_90, expected_loss, model)

    logging.debug("  - Random vectors: comparing with cosine.")
    for _ in range(20):
        streamline = [[0., 0, 0], _get_random_vector(3)]
        random_dir = _get_random_vector(3)
        # model outputs -cos(a,b), but scipy's cosine computes 1-cos(a,b)
        # then we add + 1.
        expected_loss = cosine(streamline[1], random_dir)
        _verify_loss(streamline, random_dir, expected_loss, model)


def test_l2regression_loss():
    logging.debug('Testing l2 regression loss')

    model = L2RegressionDG(input_size=1)
    streamline = [[0., 0, 0], _get_random_vector(3)]

    logging.debug("  - Identical vectors x: expecting 0")
    good_dir = streamline[1]
    expected_loss = 0.
    _verify_loss(streamline, good_dir, expected_loss, model)

    # Test for random vector, compared to scipy's euclidean
    for _ in range(20):
        wrong_dir = _get_random_vector(3)
        expected_loss = euclidean(good_dir, wrong_dir)
        _verify_loss(streamline, wrong_dir, expected_loss, model)


def test_sphere_classification_loss():
    logging.debug('Testing sphere classification loss')

    model = SphereClassificationDG(input_size=1)
    sphere = get_sphere('symmetric724')

    logging.debug("  - Neg log likelihood, expecting -ln(softmax).")

    logging.debug("      - Exactly the right class.")
    # exactly the right class (#4)
    # Note. To be realistic:
    # With as many classes (724), the value of the output must be very
    # high to have a low loss. The outputs (logits) don't have to be
    # probabilities, as a softmax will be applied by torch.
    streamline = [[0., 0, 0], sphere.vertices[4]]
    good_logit = torch.zeros(724, dtype=torch.float32)
    good_logit[4] = 100
    expected_loss = -torch.log(softmax(good_logit, dim=-1))[4]
    _verify_loss(streamline, good_logit, expected_loss, model)

    logging.debug("      - Exactly the right class test 2.")
    logit = np.random.rand(724).astype('float32')
    logit[4] = 1
    expected_loss = -torch.log(softmax(good_logit, dim=-1))[4]
    _verify_loss(streamline, good_logit, expected_loss, model)

    logging.debug("      - With epsilon difference in the target: "
                  "Should get the same class.")
    eps = 1e-3
    good_logit[29] = eps
    streamline = [[0., 0, 0], sphere.vertices[4] + eps]
    expected_loss = -torch.log(softmax(good_logit, dim=-1))[4]
    _verify_loss(streamline, good_logit, expected_loss, model)

    logging.debug("      - Random")
    logit = torch.as_tensor(np.random.rand(724).astype('float32'))
    streamline = [[0., 0, 0], sphere.vertices[4]]
    expected_loss = -torch.log(softmax(logit, dim=-1))[4]
    _verify_loss(streamline, logit, expected_loss, model)


def test_gaussian_loss():
    logging.debug('Testing gaussian loss:')

    model = SingleGaussianDG(input_size=1)

    logging.debug("      - x = mu")
    for _ in range(20):
        out_means = _get_random_vector(3)
        streamline = [[0., 0, 0], out_means]

        out_means = torch.as_tensor(out_means, dtype=torch.float32)
        out_sigmas = torch.as_tensor(np.exp(_get_random_vector(3)),
                                     dtype=torch.float32)

        # expected: x-mu = 0 ==> mahalanobis = 0
        expected_loss = -(-3 / 2 * torch.log(2 * torch.as_tensor(np.pi)) -
                          torch.log(torch.prod(out_sigmas)))
        _verify_loss(streamline, (out_means, out_sigmas), expected_loss, model)

    logging.debug("      - random")
    for _ in range(20):
        out_means = torch.as_tensor(_get_random_vector(3), dtype=torch.float32)
        out_sigmas = torch.as_tensor(np.exp(_get_random_vector(3)),
                                     dtype=torch.float32)
        b = _get_random_vector(3)
        streamline = [[0., 0, 0], b]

        # Manual logpdf computation
        b = torch.as_tensor(b, dtype=torch.float32)
        logpdf = _independent_gaussian_log_prob_vector(b, out_means, out_sigmas)
        expected_loss = -logpdf

        _verify_loss(streamline, (out_means, out_sigmas), expected_loss, model)


def test_mixture_loss():
    logging.debug('Testing mixture loss')

    model = GaussianMixtureDG(input_size=1)

    logging.debug("      - Random")
    for _ in range(20):
        # 3 Gaussians * (1 mixture param + 3 means + 3 variances)
        # (no correlations)
        out_mixture_logits = torch.as_tensor(_get_random_vector(3),
                                             dtype=torch.float32)
        out_means = torch.as_tensor(_get_random_vector(3 * 3).reshape((3, 3)),
                                    dtype=torch.float32)
        out_sigmas = torch.as_tensor(
            np.exp(_get_random_vector(3 * 3)).reshape((3, 3)),
            dtype=torch.float32)
        b = _get_random_vector(3)
        streamline = [[0., 0, 0], b]

        # Manual logpdf computation
        mixture_params = softmax(out_mixture_logits, dim=-1)
        logpdfs = torch.as_tensor([
            _independent_gaussian_log_prob_vector(b, out_means[i], out_sigmas[i])
            for i in range(3)])
        expected = -logsumexp(torch.log(mixture_params) + logpdfs, dim=-1)
        _verify_loss(streamline, (out_mixture_logits, out_means, out_sigmas),
                     expected, model)


def test_fisher_von_mises():
    logging.debug('Testing fisher-Von mises loss')

    model = FisherVonMisesDG(input_size=1)

    logging.debug("  - Expecting log prob.")

    logging.debug("      - x = mu")
    out_means = _get_random_vector(3)
    streamline = [[0., 0, 0], out_means]

    out_means = torch.as_tensor(out_means, dtype=torch.float32)
    out_kappa = torch.as_tensor(np.exp(_get_random_vector(1)),
                                dtype=torch.float32)

    expected = -fisher_von_mises_log_prob_vector(out_means, out_kappa, out_means)
    _verify_loss(streamline, (out_means, out_kappa), expected, model)

    logging.debug("      - Special case: Kappa very small")
    out_kappa = torch.as_tensor(1e-8, dtype=torch.float32)
    expected = -fisher_von_mises_log_prob_vector(out_means, out_kappa, out_means)
    _verify_loss(streamline, (out_means, out_kappa), expected, model)

    logging.debug("      - Random")
    b = _get_random_vector(3)
    streamline = [[0., 0, 0], b]

    b = torch.as_tensor(b, dtype=torch.float32)
    expected = -fisher_von_mises_log_prob_vector(out_means, out_kappa, b)
    _verify_loss(streamline, (out_means, out_kappa), expected, model)


if __name__ == '__main__':
    test_cosine_regression_loss()
    test_l2regression_loss()
    test_sphere_classification_loss()
    test_gaussian_loss()
    test_mixture_loss()
    test_fisher_von_mises()
