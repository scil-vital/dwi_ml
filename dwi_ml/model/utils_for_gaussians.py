# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.distance import mahalanobis

"""
Variables:
    C = the covariance matrix. It indicates the relations between each all
        dimensions x,y,z. C is diagonal if the axis are independant, with
        the variances on the diagonal.
    N(x) = The normal distribution
    d = the dimension of data (here, 3: x,y,z)

Formulas:
    1.  Square Mahalanobis distance for one Gaussian:
            m^2 = (x - mu)^T * inv(C) * (x - mu)

        *For independant variables:
            m^2 = sum_d [(x - mu)^2 / sigma^2]

    2. Probability function for a single multivariate Gaussian:
            P(x) =  N(x | mu, C)
                =  1 / sqrt(   det(2pi * C)  ) * exp(-0.5m^2)
        Knowing that det(kX) = k^d*det(X) where d=the size of X
                =  1 / sqrt(  (2pi)^d * det(C)   ) * exp(-0.5m^2)

        *If the variables (x,y,z axis) are independent, det(C) = sum(sigma^2).

    3. Log-likelihood of a single gaussian:
        log(P(x) = -0.5( dlog(2pi) + m^2) - sum(log(sigma))


    4. Probability function for a mixture of multivariate Gaussians
            P(X=x) = sum_k (P(X=x | Z=k)*P(Z=k)

        We note z_i = P(Z=k), the probability of observing a point that came
        from the ith Gaussian. It is actually equivalent to the mixing
        coefficient for that Gaussian.
            P(x) = sum_K (z_i *  N(x | mu_i, C_i))
                 = sum_K (z_i /sqrt(  (2pi)^d*det(C_i)  ) * exp(-0.5m_i^2))

    5. Log-likelihood for a mixture of Gaussians:
            log(P(x)) = log(sum_K(z_i/sqrt((2pi)^d*det(C_i)) * exp(-0.5m_i^2)))

        If z_i is already known (mixture_probs, computed separately), it is
        easy to separate this variable in the computation by using
        x = exp(log(x))

           log(P(x)) = log(sum_K(exp(log((z_i /sqrt((2pi)^d * det(C_i)) *
                                                             exp(-0.5m_i^2)))))
                     = ...
                     = logsumexp(log(z_i) - 0.5(d*log(2pi) + log(det(C_i)) +
                                                                        m_i^2))

        *For independant variables: det(C) = product of diagonal = sum(sigma^2)
                    = logsumexp( log(z_i) - 0.5( d*log(2pi) + m_i^2 ) -
                                                              0.5log(det(C_i)))
                    = logsumexp( log(z_i) - 0.5( d*log(2pi) + m_i^2 ) -
                                                               sum(log(sigma)))
                                          -------------------------------------
                                                 This part is the logpdf of
                                                    a single Gaussian!
                    = logsumexp( log(z_i) + logpdf_i)

   ref:
      https://en.wikipedia.org/wiki/Mahalanobis_distance
      https://stephens999.github.io/fiveMinuteStats/intro_to_em.html
      https://www.ee.columbia.edu/~stanchen/spring16/e6870/slides/lecture3.pdf
      https://github.com/jych/cle/blob/master/cle/cost/__init__.py
    """
d = 3


def independent_gaussian_log_prob_vector(x, mus, sigmas):
    """
    Parameters
    ----------
    x = the variable
    mu = mean of the gaussian (x,y,z directions)
    sigmas = standard deviation of the gaussian (x,y,z directions)
    """
    # The inverse of a diagonal matrix is just inversing values on the
    # diagonal
    cov_inv = np.eye(d) * (1 / sigmas ** 2)

    # sum(log) = log(prod)
    logpdf = -d / 2 * np.log(2 * np.pi) - np.log(np.prod(sigmas)) \
             - 0.5 * mahalanobis(x[:3], mus, cov_inv) ** 2
    return logpdf


def independent_gaussian_log_prob(targets, mus, sigmas):
    """
    This function computes the log likelihood for **individual** multivariate
    Gaussians, computed from tensors.

    If K>1 (the number of Gaussians), then the variables mu and sigma contain
    the information for many Gaussians, and log likelihoods for each Gaussian
    are returned.

    Parameters
    ----------
    targets: torch.tensor
        The variable, of shape [batch_size, none, d], d=3.
    mus: torch.tensor
        The mean, of size [batch_size, n, d], d=3.
    sigmas: torch.tensor
        The standard deviation
    """
    log_2pi = np.log(2 * np.pi).astype(np.float32)
    squared_m = ((targets - mus) / sigmas).pow(2).sum(dim=-1)
    log_det = sigmas.log().sum(dim=-1)

    gaussians_log_prob = -0.5 * (d * log_2pi + squared_m) - log_det

    return gaussians_log_prob
