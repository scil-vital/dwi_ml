.. _ref_formulas:

(Some formulas)
===============

Log-likelihood of a single Gaussian
-----------------------------------

**Variables:**

    - N(x) = The normal distribution, with mean mu and variance sigma^2.
    - C = the covariance matrix. It indicates the relations between each all dimensions x,y,z. C is diagonal if the axis are independant, with the variances on the diagonal.
    - d = the dimension of data (here, 3: x,y,z)

    We do pose that the axis are independant. WHY CAN WE POSE THAT????

**Formulas:**

    1.  Definition: The Square Mahalanobis distance for one Gaussian:

        .. math::

            m^2 = (x - \mu)^T \cdot C^{-1} \cdot (x - \mu)

        * For independant variables:

        .. math::

            m^2 = \sum_d \frac{(x - \mu)^2}{\sigma^2}

    2. Definition: Probability function for a single multivariate Gaussian:

        .. math::

            P(x) =  N(x | \mu, C)
                 =  \frac{1}{\sqrt{\det(2\pi C)}}{e^{-0.5m^2}}

        Kowing that:

        .. math::

            \det(kX) = k^d \cdot \det(X),

        We obtain

        .. math::

            P(x) =  \frac{1}{\sqrt{(2\pi)^d \cdot \det(C))}} \cdot e^{-0.5m^2}

        * For independant variables:

        .. math::

            \det(C) = \sum \sigma_i^2

        And thus we obtain:

        .. math::

            P(x) =  \frac{1}{\sqrt{(2\pi)^d \cdot \sum \sigma_i^2}} \cdot e^{-0.5m^2}


    3. Finally, we can calculate the log-likelihood of a single Gaussian:

        .. math::

            \log(P(x)) = \log(1) - \log(\sqrt{(2\pi)^d \cdot \sum \sigma_i^2}) + (-0.5m^2)

            \log(P(x)) = - 0.5\log((2\pi)^d \cdot \sum \sigma_i^2) - 0.5m^2

            \log(P(x)) = -0.5 (d\log(2\pi) + 2\log(\sigma_i) + m^2)

            \log(P(x)) = -0.5(d\log(2\pi) + m^2) - \sum \log(\sigma_i)

Log-likelihood of a mixture of Gaussians
----------------------------------------

**Variables:**

    - z_i = P(z=i), the probability that the current observation belongs to the ith Gaussian, is called the mixture coefficient.

**Formulas:**

    1. Probability function for a mixture of multivariate Gaussians

        .. math::

            P(x) = \sum_k (P(x | z=k) \cdot P(z=k)

            P(x) = \sum_k (z_k *  N(x | \mu_k, C_k))

            P(x) = \sum_k \frac{z_k}{\sqrt{(2\pi)^d \cdot \det(C_k)}} \cdot e^{-0.5m_k^2}

    2. Log-likelihood for a mixture of Gaussians:

        .. math::

            \log(P(x)) = \log(\sum_k\frac{z_k}{\sqrt{(2\pi)^d \cdot \det(C_k)}} \cdot e^{-0.5m_k^2})

        If z_i is already known (mixture coefficient, computed separately), it is easy to separate this variable in the computation by using x = exp(log(x)), giving a shape typically known as logsumexp.

        .. math::

           \log(P(x)) = \log(\sum_k(\exp(\log(\frac{z_k}{\sqrt{(2pi)^d \cdot \det(C_k)}} \cdot e^{-0.5m_k^2})))

                     = \text{logsumexp}[\log(z_k) - 0.5(d\log(2\pi) + \log(\det(C_k)) + m_k^2)]

        * For independant variables:

        .. math::

                    = \text{logsumexp}[\log(z_k) - 0.5(d\log(2\pi) + m_k^2 ) - 0.5\log(\det(C_k)]

                    = \text{logsumexp}[\log(z_k) - 0.5( d\log(2\pi) + m_k^2 ) - \sum_i(\log \sigma_i)]

        Note that the second half of this is the logpdf of a single Gaussian!

        .. math::

                    = \text{logsumexp}[\log(z_k) + \text{logpdf}_k]

References:

  - https://en.wikipedia.org/wiki/Mahalanobis_distance
  - https://stephens999.github.io/fiveMinuteStats/intro_to_em.html
  - https://www.ee.columbia.edu/~stanchen/spring16/e6870/slides/lecture3.pdf
  - https://github.com/jych/cle/blob/master/cle/cost/__init__.py

Log-likelihood of a single Fisher-Von Mises distribution
--------------------------------------------------------

**Variables:**

    - v = the normalized target
    - mu = the mean
    - kappa = the concentration parameter
    - d = the dimension
    - C = the distribution normalizing constant
    - I_n = the modified Bessel function at order n (see `wiki <https://en.wikipedia.org/wiki/Bessel_function#Modified_Bessel_functions:_I%CE%B1,_K%CE%B1>`_ or `Wolfram <https://mathworld.wolfram.com/ModifiedBesselFunctionoftheFirstKind.html>`_).

**Formulas:**

    1. Probability function:

        .. math::

            P(v | \mu, \kappa) = C e^{\kappa \cdot \mu^T \cdot v}

        Where

        .. math::

            C(\kappa) = \frac{\kappa^{\frac{d}{2}-1}}{(2\pi)^\frac{d}{2} \cdot I_{\frac{d}{2}-1}(\kappa)}

        In our case, d=3, the value is reduced to the following expression, as stated `here <https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>`_.

        .. math::

            C = \frac{\kappa}{2\pi\cdot (e^\kappa - e^{-\kappa})}

    2. log-likelihood:

        .. math::

            \log(P(v)) = \log(C e^{\kappa \cdot \mu^T \cdot v})

                       = log(C) + \kappa \cdot \mu^T \cdot v

        Where

        .. math::

            \log(C) = \log(\kappa) - \log(2\pi) - \log(e^\kappa - e^{-\kappa})

References:

  - https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
  - http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf
