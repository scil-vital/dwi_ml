.. _direction_getters:

Create a tractography model: use a DirectionGetter
==================================================

Direction getter layers should be used as last layer of any streamline generation model for the tractography task. They define the format of the output and possible associated loss functions.

General architecture
--------------------

All models have been defined as 2-layer neural networks, with the hidden layer-size the half of the input size (the input, here, is the output of the main model), and the output size depends on each model as described below. ReLu activation and dropout layers are added.

    ``input  -->  NN layer 1 --> ReLu --> dropout -->  NN layer 2 --> output``

In all cases, the mean of loss value for all points of the streamline is computed.

Each version of the DirectionGetter also adds an option to generate outputs in the form (final_direction, EOS), where the EOS is the "end-of-streamline" probability, if you want your model to learn a stopping criterion automatically.

Regression models
-----------------

These models use regression to learn directly a direction, formatted as a coordinate [x, y, z]. If EOS is used, then the direction is formatted as [x, y, z, eos], where eos is a probability between 0 and 1.

        +------------------------+---------------------------------+
        | Shape of the output    | A vector of length 3 or 4       |
        +------------------------+---------------------------------+
        | Loss function          | 3 choices, see below            |
        +------------------------+---------------------------------+
        | Deterministic tracking | Direct use of the direction     |
        +------------------------+---------------------------------+
        | Probabilistic tracking | Impossible                      |
        +------------------------+---------------------------------+

Three models are possible, for three options of loss function:

- **CosineRegressionDG**: The loss is the cosine similarity between the computed direction y and the provided target direction t.

    .. math::

        cos(\theta) = \frac{y \cdot t}{\|y\| \|t\|}

- **L2RegressionDG**: The loss is the pairwise distance between the computed direction y and the provided target direction t.

    .. math::
        \sqrt{\sum(t_i - y_i)^2}

- **CosPlusL2RegressionDG**: The loss is the sum of both of the above.

Classification models
---------------------

This model uses classification by formatting directions as a choice of direction among a list of discrete points on the sphere (Ex: ``dipy.data.get_sphere('symmetric724')``). Each point is a class. If EOS is used, it represents an additional class.


        +------------------------+----------------------------------+
        | Shape of the output    | A vector of length K (nb class)  |
        +------------------------+----------------------------------+
        | Loss function          | 2 choices, see below             |
        +------------------------+----------------------------------+
        | Deterministic tracking | Class with higher probability    |
        +------------------------+----------------------------------+
        | Probabilistic tracking | Sampled from class probabilities |
        +------------------------+----------------------------------+

- **SphereClassificationDG**: The loss is the negative log-likelyhood from a softmax (integrated in torch) (equivalent to the cross-entropy).

- **SmoothSphereClassificationDG**: This model uses the same classes, but the targets are not represented as one-hot vectors. Instead, directions close to the target class have a non-null probability. See `Benou et al., 2019 <https://link.springer.com/chapter/10.1007/978-3-030-32248-9_70>`_.


Models for indirect regression
------------------------------

Gaussian models
'''''''''''''''

This is a regression model that learns parameters representing a *probability function* for a Gaussian [1]_ [2]_. The model learns to represent the mean and variance of the gaussian functions that could represent each data and its uncertainty. See :ref:`ref_formulas` for the complete formulas.

.. NOTE::
    Do not counfound this with the tensor, which is also a multivariate Gaussian. Here, the 3D (x, y, z) means of the Gaussian represent the probable next direction, and the variance represent the uncertainty in each axis. Whereas in a tensor, the Gaussian represent the direction of anisotropy.

- **SingleGaussianDG**: the model is a 2-layer NN for the means and a 2-layer NN for the variances. The output is 6 parameters: 3 means (x, y, z) and 3 variances (x, y, z). If EOS is used, it is a 7th learned value.

        +------------------------+-------------------------------------------------+
        | Shape of the output    | A vector of length 6 or 7                       |
        +------------------------+-------------------------------------------------+
        | Loss function          | Negative log-likelihood.                        |
        +------------------------+-------------------------------------------------+
        | Deterministic tracking | The direction is directly given by the mean.    |
        +------------------------+-------------------------------------------------+
        | Probabilistic tracking | Sampled from the 3D distribution.               |
        +------------------------+-------------------------------------------------+

- **GaussianMixtureDG**: In this case, the models learns to represent the function probability as a mixture of N Gaussians, possibly representing direction choices in the case of fiber crossing and other special configurations. The loss is again the negative log-likelihood. Note that the model is a 2-layer NN for the mean and a 2-layer NN for the variance, for each of N Gaussians. The output is N * (6 parameters: 3 means (x, y, z) and 3 variances (x, y, z) plus a mixture parameter for each, giving the probability that the right direction would be given by this Gaussian.

        +------------------------+-------------------------------------------------+
        | Shape of the output    | A vector of length N * (7 or 8)                 |
        +------------------------+-------------------------------------------------+
        | Loss function          | Negative log-likelihood.                        |
        +------------------------+-------------------------------------------------+
        | Deterministic tracking | Mean of the most probable gaussian.             |
        +------------------------+-------------------------------------------------+
        | Probabilistic tracking | Sampled from the multi-3D distribution.         |
        +------------------------+-------------------------------------------------+

Note that tyically, in the literature, Gaussian mixtures are used with expectation-maximisation (EM). Here we simply update the mixture parameters and the Gaussian parameters jointly, similar to GMM in https://github.com/jych/cle/blob/master/cle/cost/__init__.py. See the detailed mathematics in :ref:`ref_formulas`.

%%%%%%%%%%%%%%%%%%%

.. [1] Some code is available `in this blog <See for example here https://blog.dominodatalab.com/fitting-gaussian-process-models-python/>`_.
.. [2] See the `Tractoinferno paper <https://www.nature.com/articles/s41597-022-01833-1>`_



Fisher von mises models
'''''''''''''''''''''''

Similarly to Gaussian models, this is a regression model that learns the distribution probability of the data. This model uses the Fisher - von Mises distribution, which resembles a gaussian on the sphere (`ref1 <https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>`_, `ref2 <http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf>`_ . As such, it does not require unit normalization when sampling, and should be more stable while training. The loss is again the negative log-likelihood. Note that the model is a 2-layer NN for the mean and a 2-layer NN for the 'kappa'. Larger kappa leads to a more concentrated cluster of points, similar to sigma for Gaussians.

See the detailed mathematics in :ref:`ref_formulas`.

- **FisherVonMisesDG**: The loss is the negative log-likelihood. Note that the model is a 2-layer NN for the means and a 2-layer NN for the variances. See :ref:`ref_formulas` for the complete formulas. The output is 4 parameters: 3 for the means and one for kappa. If EOS is used, it is a 5th learned value.


        +------------------------+-------------------------------------------------+
        | Shape of the output    | A vector of length 4 or 5.                      |
        +------------------------+-------------------------------------------------+
        | Loss function          | Negative log-likelihood.                        |
        +------------------------+-------------------------------------------------+
        | Deterministic tracking | The means                                       |
        +------------------------+-------------------------------------------------+
        | Probabilistic tracking | Sampled from the rejection sampling [3]_        |
        +------------------------+-------------------------------------------------+

**FisherVonMisesMixtureDG**: Not implemented yet.

Other ideas
'''''''''''

An equivalent model could learn to represent the direction on the sphere to learn a normalized direction. The means would be 2D (phi, rho), and the variances too. This has not been implemented yet.


%%%%%%%%%%%%%%%%%%%

.. [3] Directional Statistics (Mardia and Jupp, 1999)), implemented `here <https://github.com/jasonlaska/spherecluster>`_.
