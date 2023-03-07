#!/usr/bin/env python
import logging

import numpy as np
from dipy.data import get_sphere
from scipy.spatial.distance import (cosine, euclidean, mahalanobis)
import torch
from torch.nn.utils.rnn import PackedSequence
from torch.nn.functional import softmax

from dwi_ml.models.direction_getter_models import (
    CosineRegressionDirectionGetter, FisherVonMisesDirectionGetter,
    GaussianMixtureDirectionGetter, L2RegressionDirectionGetter,
    SingleGaussianDirectionGetter, SphereClassificationDirectionGetter)
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

tol = 1e-5
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
    logpdf = -d / 2 * torch.log(2 * torch.pi) - torch.log(torch.prod(sigmas)) \
             - 0.5 * mahalanobis(x[:3], mus, cov_inv) ** 2
    return logpdf


def _get_random_vector(size=3):
    scaling = np.random.randint(1, 9)
    return torch.as_tensor(np.random.randn(size),
                           dtype=torch.float32) * scaling


def _prepare_packedsequence(a):
    if not isinstance(a, PackedSequence):
        a = PackedSequence(data=(torch.as_tensor(a[None, :],
                                                 dtype=torch.float32)),
                           batch_sizes=torch.as_tensor([1]))
        return a


def _verify_loss(streamline, fake_model_output, expected_loss, model,
                 expected_eos_loss=torch.as_tensor(0.)):
    streamline = torch.as_tensor(streamline, dtype=torch.float32)
    targets = model.prepare_targets([streamline])[0]
    fake_model_output = torch.as_tensor(fake_model_output, dtype=torch.float32)
    expected_loss = torch.as_tensor(expected_loss, dtype=torch.float32)

    computed_loss, _ = model.compute_loss(fake_model_output, targets)
    computed_loss, loss_eos = computed_loss

    assert torch.allclose(computed_loss, expected_loss, atol=tol), \
        "Expected loss {} but got {}.\n" \
        "      - Streamline: {}    ( = dir {})\n" \
        "      - Fake model output: {}\n" \
        .format(expected_loss, computed_loss,
                streamline, streamline[1, :] - streamline[0, :],
                fake_model_output)
    assert torch.allclose(loss_eos, expected_eos_loss), \
        "Expected EOS loss {} but got {}".format(expected_eos_loss, loss_eos)


def test_cosine_regression_loss():
    logging.debug('Testing cosine regression loss')

    model = CosineRegressionDirectionGetter(3)
    streamline = [[0., 0, 0], [1., 0, 0]]

    logging.debug("  - Identical vectors x: expecting -1")
    good_dir = [1., 0, 0]
    expected_loss = -1.
    _verify_loss(streamline, good_dir, expected_loss, model)

    logging.debug("  - Vectors with same angle: expecting -1")
    scales = np.random.random(20) * 20
    for s in scales:
        good_dir_bad_scale = [s, 0., 0]
        expected_loss = -1.
        _verify_loss(streamline, good_dir_bad_scale, expected_loss, model)

    logging.debug("  - Vectors with at 90 degrees 1: expecting 0")
    dir_90 = [0., 1, 0]
    expected_loss = 0.
    _verify_loss(streamline, dir_90, expected_loss, model)

    logging.debug("  - Vectors with at 180 degrees: expecting 1")
    dir_90 = [-1., 0, 0]
    expected_loss = 1.
    _verify_loss(streamline, dir_90, expected_loss, model)

    logging.debug("  - Random vectors: comparing with cosine.")
    for _ in range(20):
        streamline = [[0., 0, 0], _get_random_vector(3)]
        random_dir = _get_random_vector(3)
        # model outputs -cos(a,b), but scipy's cosine computes 1-cos(a,b)
        expected_loss = cosine(streamline[1], random_dir) - 1
        _verify_loss(streamline, random_dir, expected_loss, model)


def test_l2regression_loss():
    logging.debug('Testing l2 regression loss')

    model = L2RegressionDirectionGetter(1)
    streamline = [[0., 0, 0], _get_random_vector(3)]

    logging.debug("  - Identical vectors x: expecting 0")
    good_dir = streamline[1]
    expected_loss = 0.
    _verify_loss(streamline, good_dir, expected_loss, model)

    # Test for random vector, compared to scipy's euclidean
    for _ in range(200):
        wrong_dir = _get_random_vector(3)
        expected_loss = euclidean(good_dir, wrong_dir)
        _verify_loss(streamline, wrong_dir, expected_loss, model)


def test_sphere_classification_loss():
    logging.debug('\nTesting sphere classification loss')

    model = SphereClassificationDirectionGetter(1)
    sphere = get_sphere('symmetric724')

    logging.debug("  - Neg log likelihood, expecting -ln(softmax).")

    logging.debug("      - Exactly the right class.")
    # exactly the right class (#4)
    # Note. To be realistic:
    # With as many classes (724), the value of the output must be very
    # high to have a low loss. The outputs (logits) don't have to be
    # probabilities, as a softmax will be applied by torch.
    streamline = [[0., 0, 0], sphere.vertices[4]]
    good_logit = torch.zeros((1, 724), dtype=torch.float32)
    good_logit[0, 4] = 100
    expected_loss = -torch.log(softmax(good_logit))[0, 1]
    _verify_loss(streamline, good_logit, expected_loss, model)

    logging.debug("      - With eps difference in the target: "
                  "Should get the same class.")
    logit = torch.zeros((1, 724)).astype('float32')
    logit[0, 1] = 1
    eps = 1e-3
    b = sphere.vertices[1] + eps
    expected = -torch.log(torch.softmax(logit))[0, 1]
    value, _ = model.compute_loss(logit, b)
    assert torch.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Exactly the right class test 2.")
    logit = np.random.rand(1, 724).astype('float32')
    logit[0, 1] = 1
    b = sphere.vertices[1]
    expected = -torch.log(softmax(logit))[0, 1]
    value, _ = model.compute_loss(logit, b)
    assert torch.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Random")
    logit = np.random.rand(1, 724).astype('float32')
    b = sphere.vertices[1]
    expected = -torch.log(softmax(logit))[0, 1]
    value, _ = model.compute_loss(logit, b)
    assert torch.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)


def test_gaussian_loss():
    logging.debug('\nTesting gaussian loss')

    np.random.seed(1234)
    model = SingleGaussianDirectionGetter(1)

    logging.debug("  - Expecting mahalanobis value")

    logging.debug("      - x = mu")
    for _ in range(20):
        a_means = _get_random_vector(3)
        a_sigmas = np.exp(_get_random_vector(3))
        b = a_means

        # expected: x-mu = 0 ==> mahalanobis = 0
        expected = -(-3 / 2 * torch.log(2 * np.pi) - torch.log(np.prod(a_sigmas)))

        value, _ = model.compute_loss((a_means, a_sigmas), b)
        assert torch.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - random")
    for _ in range(200):
        a_means = _get_random_vector(3)
        a_sigmas = np.exp(_get_random_vector(3))
        b = _get_random_vector(3)

        # Manual logpdf computation
        logpdf = _independent_gaussian_log_prob_vector(b, a_means, a_sigmas)
        expected = -logpdf

        value, _ = model.compute_loss((a_means, a_sigmas), b)
        assert torch.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_mixture_loss():
    logging.debug('\nTesting mixture loss')

    np.random.seed(1234)
    model = GaussianMixtureDirectionGetter(1)

    logging.debug("  - Expecting neg logsumexp(log_mixture + logpdf)")

    logging.debug("      - Random")
    for _ in range(200):
        # 3 Gaussians * (1 mixture param + 3 means + 3 variances)
        # (no correlations)
        a_mixture_logits = _get_random_vector(3)
        a_means = _get_random_vector(3 * 3).reshape((3, 3))
        a_sigmas = np.exp(_get_random_vector(3 * 3)).reshape((3, 3))
        b = _get_random_vector(3)

        # Manual logpdf computation
        mixture_params = softmax(a_mixture_logits)
        logpdfs = torch.as_tensor(
            [_independent_gaussian_log_prob_vector(b, a_means[i], a_sigmas[i])
             for i in range(3)])
        expected = -logsumexp(torch.log(mixture_params) + logpdfs)

        value, _ = model.compute_loss(
            (a_mixture_logits, a_means, a_sigmas), b)
        assert torch.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_fisher_von_mises():
    logging.debug('\nTesting fisher-Von mises loss')

    model = FisherVonMisesDirectionGetter(1)

    logging.debug("  - Expecting log prob.")

    logging.debug("      - x = mu")
    a_means = _get_random_vector(3)
    a_kappa = np.exp(_get_random_vector(1))
    b = a_means

    expected = -fisher_von_mises_log_prob_vector(a_means, a_kappa, b)
    value, _ = model.compute_loss((a_means, a_kappa), b)
    assert torch.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Random")
    a_means = _get_random_vector(3)
    a_kappa = np.exp(_get_random_vector(1))
    b = _get_random_vector(3)

    expected = -fisher_von_mises_log_prob_vector(a_means, a_kappa, b)
    value, _ = model.compute_loss((a_means, a_kappa), b)
    assert torch.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)


if __name__ == '__main__':
    test_cosine_regression_loss()
    test_l2regression_loss()
    test_sphere_classification_loss()
    test_gaussian_loss()
    test_mixture_loss()
    test_fisher_von_mises()
