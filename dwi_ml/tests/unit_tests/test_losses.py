#!/usr/bin/env python
import logging

from dipy.data import get_sphere
from nose.tools import assert_equal
import numpy as np
from scipy.spatial.distance import (cosine, euclidean, mahalanobis)
from scipy.special import logsumexp, softmax
import torch
from torch.nn.utils.rnn import PackedSequence

from dwi_ml.models.direction_getter_layers import (
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
    cov_inv = np.eye(d) * (1 / sigmas ** 2)

    # sum(log) = log(prod)
    logpdf = -d / 2 * np.log(2 * np.pi) - np.log(np.prod(sigmas)) \
             - 0.5 * mahalanobis(x[:3], mus, cov_inv) ** 2
    return logpdf


def _get_random_vector(size=3):
    scaling = np.random.randint(1, 10)
    return np.array(np.random.randn(size), dtype=np.float32) * scaling


def _prepare_tensor(a):
    if isinstance(a, tuple):
        a = tuple([torch.as_tensor(i[None, :], dtype=torch.float32)
                   for i in a])
    elif isinstance(a, np.ndarray):
        a = torch.as_tensor(a[None, :], dtype=torch.float32)
    return a


def _prepare_packedsequence(a):
    if not isinstance(a, PackedSequence):
        a = PackedSequence(data=(torch.as_tensor(a[None, :],
                                                 dtype=torch.float32)),
                           batch_sizes=torch.as_tensor([1]))
        return a


def _compute_loss_tensor(outputs, targets, model):
    outputs = _prepare_tensor(outputs)
    targets = _prepare_tensor(targets)

    mean_loss, _ = model.compute_loss(outputs, targets)
    # logging.debug("Means loss: {}.".format(mean_loss))

    return np.asarray(mean_loss)


def test_cosine_regression_loss():
    logging.debug('Testing cosine regression loss')

    np.random.seed(1234)
    model = CosineRegressionDirectionGetter(3)

    logging.debug("  - Identical vectors x: expecting -1")
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    expected = np.array(-1)
    value = _compute_loss_tensor(a, b, model)
    assert_equal(value, expected)

    logging.debug("  - Identical vectors y: expecting -1")
    a = np.array([0, 1, 0])
    b = np.array([0, 1, 0])
    expected = np.array(-1)
    value = _compute_loss_tensor(a, b, model)
    assert_equal(value, expected)

    logging.debug("  - Identical vectors z: expecting -1")
    a = np.array([0, 0, 1])
    b = np.array([0, 0, 1])
    expected = np.array(-1)
    value = _compute_loss_tensor(a, b, model)
    assert_equal(value, expected)

    logging.debug("  - Vectors with same angle: expecting -1")
    scales = np.random.random(20) * 20
    for s in scales:
        a = np.array([1, 0, 0])
        b = a * s
        expected = np.array(-1)
        value = _compute_loss_tensor(a, b, model)
        assert_equal(value, expected)

    logging.debug("  - Vectors with at 90 degrees 1: expecting 0")
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    expected = np.array(0)
    value = _compute_loss_tensor(a, b, model)
    assert_equal(value, expected)

    logging.debug("  - Vectors with at 90 degrees 2: expecting 0")
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 1])
    expected = np.array(0)
    value = _compute_loss_tensor(a, b, model)
    assert_equal(value, expected)

    logging.debug("  - Vectors with at 90 degrees random: expecting 0")
    for _ in range(20):
        a = _get_random_vector(3)
        b = _get_random_vector(3)
        c = np.cross(a, b)
        expected = np.array(0)

        value = _compute_loss_tensor(a, c, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

        value = _compute_loss_tensor(b, c, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("  - Vectors with at 180 degrees random: expecting 1")
    for _ in range(20):
        a = _get_random_vector(3)
        b = np.array(-a * (np.random.random() + 1e-3) *
                     np.random.randint(1, 10), dtype=np.float32)
        expected = np.array(1)

        value = _compute_loss_tensor(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("  - Random vectors: comparing with cosine.")
    for _ in range(200):
        a = _get_random_vector(3)
        b = _get_random_vector(3)
        # model outputs -cos(a,b), but cosine computes 1-cos(a,b)
        expected = cosine(a, b) - 1

        value = _compute_loss_tensor(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_l2regression_loss():
    logging.debug('\nTesting l2 regression loss')

    np.random.seed(1234)
    model = L2RegressionDirectionGetter(1)

    logging.debug("  - Identical vectors: expecting 0")
    a = _get_random_vector(3)
    b = a
    expected = np.array(0)
    value = _compute_loss_tensor(a, b, model)
    assert np.allclose(value, expected, atol=tol),\
        "Failed; got: {}; expected: {}".format(value, expected)

    # Test for random vector, compared to scipy's euclidean
    for _ in range(200):
        a = _get_random_vector(3)
        b = _get_random_vector(3)
        expected = euclidean(a, b)
        value = _compute_loss_tensor(a, b, model)
        assert np.allclose(value, expected, atol=tol),\
            "Failed; got: {}; expected: {}".format(value, expected)


def test_sphere_classification_loss():
    logging.debug('\nTesting sphere classification loss')

    model = SphereClassificationDirectionGetter(1)
    sphere = get_sphere('symmetric724')

    logging.debug("  - Neg log likelihood, expecting -ln(softmax).")

    logging.debug("      - Exactly the right class.")
    # exactly the right class (#1)
    # Note. To be realistic:
    # With as many classes (724), the value of the output must be very
    # high to have a low loss. The outputs (logits) don't have to be
    # probabilities, as a softmax will be applied by torch.
    logit = np.zeros((1, 724)).astype('float32')
    logit[0, 1] = 100
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = _compute_loss_tensor(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - With eps difference in the target: "
                  "Should get the same class.")
    logit = np.zeros((1, 724)).astype('float32')
    logit[0, 1] = 1
    eps = 1e-3
    b = sphere.vertices[1] + eps
    expected = -np.log(softmax(logit))[0, 1]
    value = _compute_loss_tensor(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Exactly the right class test 2.")
    logit = np.random.rand(1, 724).astype('float32')
    logit[0, 1] = 1
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = _compute_loss_tensor(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Random")
    logit = np.random.rand(1, 724).astype('float32')
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = _compute_loss_tensor(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
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
        expected = -(-3 / 2 * np.log(2 * np.pi) - np.log(np.prod(a_sigmas)))

        value = _compute_loss_tensor((a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - random")
    for _ in range(200):
        a_means = _get_random_vector(3)
        a_sigmas = np.exp(_get_random_vector(3))
        b = _get_random_vector(3)

        # Manual logpdf computation
        logpdf = _independent_gaussian_log_prob_vector(b, a_means, a_sigmas)
        expected = -logpdf

        value = _compute_loss_tensor((a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
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
        logpdfs = np.array(
            [_independent_gaussian_log_prob_vector(b, a_means[i], a_sigmas[i])
             for i in range(3)])
        expected = -logsumexp(np.log(mixture_params) + logpdfs)

        value = _compute_loss_tensor(
            (a_mixture_logits, a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
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
    value = _compute_loss_tensor((a_means, a_kappa), b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    logging.debug("      - Random")
    a_means = _get_random_vector(3)
    a_kappa = np.exp(_get_random_vector(1))
    b = _get_random_vector(3)

    expected = -fisher_von_mises_log_prob_vector(a_means, a_kappa, b)
    value = _compute_loss_tensor((a_means, a_kappa), b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)


if __name__ == '__main__':
    test_cosine_regression_loss()
    test_fisher_von_mises()
    test_gaussian_loss()
    test_l2regression_loss()
    test_mixture_loss()
    test_sphere_classification_loss()
