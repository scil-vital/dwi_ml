from dipy.data import get_sphere
from dwi_ml.model.direction_getter_models import (
    CosineRegressionDirectionGetter, FisherVonMisesDirectionGetter,
    GaussianMixtureDirectionGetter, L2RegressionDirectionGetter,
    SingleGaussianDirectionGetter, SphereClassificationDirectionGetter)
from dwi_ml.model.utils_for_gaussians import (
    independent_gaussian_log_prob_vector)
from dwi_ml.model.utils_for_fisher_von_mises import (
    fisher_von_mises_log_prob_vector)
from nose.tools import assert_equal
import numpy as np
from scipy.spatial.distance import (cosine, euclidean)
from scipy.special import logsumexp, softmax
import torch
from torch.nn.utils.rnn import PackedSequence


"""
Included tests are:
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


def get_random_vector(size=3):
    scaling = np.random.randint(1, 10)
    return np.array(np.random.randn(size), dtype=np.float32) * scaling


def prepare_tensor(a):
    if isinstance(a, tuple):
        a = tuple([torch.as_tensor(i[None, :], dtype=torch.float32)
                   for i in a])
    elif isinstance(a, np.ndarray):
        a = torch.as_tensor(a[None, :], dtype=torch.float32)
    return a


def prepare_packedsequence(a):
    if not isinstance(a, PackedSequence):
        a = PackedSequence(data=(torch.as_tensor(a[None, :],
                                                 dtype=torch.float32)),
                           batch_sizes=torch.as_tensor([1]))
        return a


def prepare_tensor_and_compute_loss(outputs, targets, model):
    outputs = prepare_tensor(outputs)
    targets = prepare_tensor(targets)
    mean_loss = model.run_model_and_compute_loss(outputs, targets).item()

    return mean_loss


def prepare_packedsequence_and_compute_loss(outputs, targets, model):
    outputs = prepare_tensor(outputs)
    targets = prepare_packedsequence(targets)
    mean_loss = model.run_model_and_compute_loss(outputs, targets.data).item()

    return mean_loss


def test_cosine_regression_loss():
    np.random.seed(1234)
    model = CosineRegressionDirectionGetter(1)

    # Test identical vectors
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 0])
    expected = np.array(-1)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert_equal(value, expected)

    # Test identical vectors
    a = np.array([0, 1, 0])
    b = np.array([0, 1, 0])
    expected = np.array(-1)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert_equal(value, expected)

    # Test identical vectors
    a = np.array([0, 0, 1])
    b = np.array([0, 0, 1])
    expected = np.array(-1)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert_equal(value, expected)

    # Test vectors with same angle
    scales = np.random.random(20) * 20
    for s in scales:
        a = np.array([1, 0, 0])
        b = a * s
        expected = np.array(-1)
        value = prepare_tensor_and_compute_loss(a, b, model)
        assert_equal(value, expected)

    # Test vectors at 90 degrees
    a = np.array([1, 0, 0])
    b = np.array([0, 1, 0])
    expected = np.array(0)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert_equal(value, expected)

    # Test vectors at 90 degrees
    a = np.array([1, 0, 0])
    b = np.array([0, 0, 1])
    expected = np.array(0)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert_equal(value, expected)

    # Test vectors at 90 degrees
    for _ in range(20):
        a = get_random_vector(3)
        b = get_random_vector(3)
        c = np.cross(a, b)
        expected = np.array(0)

        value = prepare_tensor_and_compute_loss(a, c, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

        value = prepare_tensor_and_compute_loss(b, c, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    # Test vectors at 180 degrees
    for _ in range(20):
        a = get_random_vector(3)
        b = np.array(-a * (np.random.random() + 1e-3) *
                     np.random.randint(1, 10), dtype=np.float32)
        expected = np.array(1)

        value = prepare_tensor_and_compute_loss(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    # Test against scipy cosine distance
    for _ in range(200):
        a = get_random_vector(3)
        b = get_random_vector(3)
        # model outputs -cos(a,b), but cosine computes 1-cos(a,b)
        expected = cosine(a, b) - 1

        value = prepare_tensor_and_compute_loss(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

        value = prepare_packedsequence_and_compute_loss(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_l2regression_loss():
    np.random.seed(1234)
    model = L2RegressionDirectionGetter(1)

    # Test for x == y
    a = get_random_vector(3)
    b = a
    expected = np.array(0)
    value = prepare_tensor_and_compute_loss(a, b, model)
    assert np.allclose(value, expected, atol=tol),\
        "Failed; got: {}; expected: {}".format(value, expected)

    # Test for random vector, compared to scipy's euclidean
    for _ in range(200):
        a = get_random_vector(3)
        b = get_random_vector(3)
        expected = euclidean(a, b)
        value = prepare_tensor_and_compute_loss(a, b, model)
        assert np.allclose(value, expected, atol=tol),\
            "Failed; got: {}; expected: {}".format(value, expected)

        value = prepare_packedsequence_and_compute_loss(a, b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_sphere_classification_loss():
    model = SphereClassificationDirectionGetter(1)
    sphere = get_sphere('symmetric724')

    # exactly the right class (#1)
    # Note. To be realistic:
    # With as many classes (724), the value of the output must be very
    # high to have a low loss. The outputs (logits) don't have to be
    # probabilities, as a softmax will be applied by torch.
    logit = np.zeros((1, 724)).astype('float32')
    logit[0, 1] = 100
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = prepare_tensor_and_compute_loss(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    # Modifying just a little the target (#1)
    logit = np.zeros((1, 724)).astype('float32')
    logit[0, 1] = 1
    eps = 1e-3
    b = sphere.vertices[1] + eps
    expected = -np.log(softmax(logit))[0, 1]
    value = prepare_tensor_and_compute_loss(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    # Modifying the logit output (still #1 the highest)
    logit = np.random.rand(1, 724).astype('float32')
    logit[0, 1] = 1
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = prepare_tensor_and_compute_loss(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    # Random logits
    logit = np.random.rand(1, 724).astype('float32')
    b = sphere.vertices[1]
    expected = -np.log(softmax(logit))[0, 1]
    value = prepare_tensor_and_compute_loss(logit, b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)


def test_gaussian_loss():
    np.random.seed(1234)
    model = SingleGaussianDirectionGetter(1)

    # Test x == \mu
    for _ in range(20):
        a_means = get_random_vector(3)
        a_sigmas = np.exp(get_random_vector(3))
        b = a_means

        # expected: x-mu = 0 ==> mahalanobis = 0
        expected = -(-3 / 2 * np.log(2 * np.pi) - np.log(np.prod(a_sigmas)))

        value = prepare_tensor_and_compute_loss((a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

    # Random tests
    for _ in range(200):
        a_means = get_random_vector(3)
        a_sigmas = np.exp(get_random_vector(3))
        b = get_random_vector(3)

        # Manual logpdf computation
        logpdf = independent_gaussian_log_prob_vector(b, a_means, a_sigmas)
        expected = -logpdf

        value = prepare_tensor_and_compute_loss((a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

        value = prepare_packedsequence_and_compute_loss((a_means, a_sigmas),
                                                        b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_mixture_loss():
    np.random.seed(1234)
    model = GaussianMixtureDirectionGetter(1)

    # Test against scipy
    for _ in range(200):
        # 3 Gaussians * (1 mixture param + 3 means + 3 variances)
        # (no correlations)
        a_mixture_logits = get_random_vector(3)
        a_means = get_random_vector(3 * 3).reshape((3, 3))
        a_sigmas = np.exp(get_random_vector(3 * 3)).reshape((3, 3))
        b = get_random_vector(3)

        # Manual logpdf computation
        mixture_params = softmax(a_mixture_logits)
        logpdfs = np.array([independent_gaussian_log_prob_vector(b, a_means[i],
                                                                 a_sigmas[i])
                            for i in range(3)])
        expected = -logsumexp(np.log(mixture_params) + logpdfs)

        value = prepare_tensor_and_compute_loss(
            (a_mixture_logits, a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)

        value = prepare_packedsequence_and_compute_loss(
            (a_mixture_logits, a_means, a_sigmas), b, model)
        assert np.allclose(value, expected, atol=tol), \
            "Failed; got: {}; expected: {}".format(value, expected)


def test_fisher_von_mises():
    model = FisherVonMisesDirectionGetter(1)

    # Test x == \mu
    a_means = get_random_vector(3)
    a_kappa = np.exp(get_random_vector(1))
    b = a_means

    expected = -fisher_von_mises_log_prob_vector(a_means, a_kappa, b)
    value = prepare_tensor_and_compute_loss((a_means, a_kappa), b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)

    # Test random
    a_means = get_random_vector(3)
    a_kappa = np.exp(get_random_vector(1))
    b = get_random_vector(3)

    expected = -fisher_von_mises_log_prob_vector(a_means, a_kappa, b)
    value = prepare_tensor_and_compute_loss((a_means, a_kappa), b, model)
    assert np.allclose(value, expected, atol=tol), \
        "Failed; got: {}; expected: {}".format(value, expected)


if __name__ == '__main__':

    print('Testing cosine regression loss')
    test_cosine_regression_loss()

    print('Testing l2 regression loss')
    test_l2regression_loss()

    print('Testing sphere classification loss')
    test_sphere_classification_loss()

    print('Testing gaussian loss')
    test_gaussian_loss()

    print('Testing mixture loss')
    test_mixture_loss()

    print('Testing fisher-Von mises loss')
    test_fisher_von_mises()
