5. Preparing your model
=======================

.. role:: underline
    :class: underline

5.1. Main models
----------------

You should make your model a child class of our **MainModelAbstract** to keep some important properties (ex, experiment name, neighborhood definition). Also, methods to save and load the model parameters on disk have been prepared.

The compute_loss method should be implemented to be used with our trainer.

For generative models, the get_tracking_direction_det and sample_tracking_direction_prob methods should be implemented to be used with our tracker.

We have also prepared child classes to help formatting previous directions, useful both for training and tracking.


5.2. Direction getter models
----------------------------

Direction getter (sub)-models should be used as last layer of any streamline generation model. They define the format of the output and possible associated loss functions.

All models have been defined as 2-layer neural networks, with the hidden layer-size the half of the input size (the input, here, is the output of the main model), and the output size depends on each model as described below. ReLu activation and dropout layers are added. Final models are as below:

            input  -->  NN layer 1 --> ReLu --> dropout -->  NN layer 2 --> output

In all cases, the mean of loss values for each timestep of the streamline is computed.

Regression models
''''''''''''''''''

Simple regression to learn directly a direction [x,y,z].

- :underline:`Shape of the output`: 3 parameters: the direction x, y, z.
- :underline:`Deterministic tracking`: Direct use of the direction.
- :underline:`Probabilistic tracking`: Impossible.

Two models for two loss functions:

- **CosineRegressionDirectionGetter**: The loss is the cosine similarity between the computed direction y and the provided target direction t.

    .. math::

        cos(\theta) = \frac{y \cdot t}{\|y\| \|t\|}

- **L2RegressionDirectionGetter**: The loss is the pairwise distance between the computed direction y and the provided target direction t.

    .. math::
        \sqrt{\sum(t_i - y_i)^2}


Classification models
'''''''''''''''''''''

- :underline:`Shape of the output`: a probability for each of K classes, corresponding to each discrete points on the sphere (Ex: `dipy.data.get_sphere('symmetric724')`)
- :underline:`Deterministic tracking`: The class (corresponding to the direction) with highest probability is chosen.
- :underline:`Probabilistic tracking`: One class (corresponding to one direction) is sampled randomly based on each class's probability.

One implemented model:

- **SphereClassificationDirectionGetter**: The loss is the negative log-likelyhood from a softmax (integrated in torch) (equivalent to the cross-entropy).

Gaussian models
'''''''''''''''

This is a regression model, but contrary to typical regression, which would learn to set the weights that would represent a function h such that y ~ h(x), gaussian processes learn directly the *probability function*. See for example here https://blog.dominodatalab.com/fitting-gaussian-process-models-python/. The model learns to represent the mean and variance of the gaussian functions that could represent each data and its uncertainty. See :ref:`ref_formulas` for the complete formulas.

To not counfound with the tensor, which is also a multivariate Gaussian. Here, the (3D, x, y, z) means of the Gaussian represent the probable next direction, and the variance represent uncertainty in each axis, whereas in a tensor, the means would represent the origin of the tensor, i.e. (0,0,0), and the variances represent the shape of the tensor in each axis.

An equivalent model could learn to represent the direction on the sphere to learn a normalized direction. The means would be 2D (phi, rho) and the variances too. This has not been implemented yet. The reason for choosing a 3D model is that users could want to work with unnormalized direction and variable step sizes, for instance to reproduce compressed streamlines.

- **SingleGaussianDirectionGetter**: The loss is the negative log-likelihood. Note that the model is a 2-layer NN for the means and a 2-layer NN for the variances.

    - :underline:`Shape of the output`: 6 parameters: 3 means (x, y, z) and 3 variances (x, y, z).
    - :underline:`Deterministic tracking`: The direction is directly given by the mean.
    - :underline:`Probabilistic tracking`: A direction is sampled from the 3D distribution.

- **GaussianMixtureDirectionGetter**: In this case, the models learns to represent the function probability as a mixture of N Gaussians, possibly representing direction choices in the case of fiber crossing and other special configurations. The loss is again the negative log-likelihood. Note that the model is a 2-layer NN for the mean and a 2-layer NN for the variance, for each of N Gaussians.

    - :underline:`Shape of the output`: N * (6 parameters: 3 means (x, y, z) and 3 variances (x, y, z) plus a mixture parameter), where N is the number of Gaussians.
    - :underline:`Deterministic tracking`: The direction is directly given as the mean of the most probable Gaussian; the one with biggest mixture coefficient.
    - :underline:`Probabilistic tracking`: A direction is sampled from the 3D distribution.

Note that tyically, in the literature, Gaussian mixtures are used with expectation-maximisation (EM). Here we simply update the mixture parameters and the Gaussian parameters jointly, similar to GMM in https://github.com/jych/cle/blob/master/cle/cost/__init__.py.

Fisher von mises models
'''''''''''''''''''''''

Similarly to Gaussian models, this is a regression model that learns the distribution probability of the data. This model uses the Fisher - von Mises distribution, which resembles a gaussian on the sphere (`ref1 <https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution>`_, `ref2 <http://www.mitsuba-renderer.org/~wenzel/files/vmf.pdf>`_ . As such, it does not require unit normalization when sampling, and should be more stable while training. The loss is again the negative log-likelihood. Note that the model is a 2-layer NN for the mean and a 2-layer NN for the 'kappas'. Larger kappa leads to a more concentrated cluster of points, similar to sigma for Gaussians.

- **FisherVonMisesDirectionGetter**: The loss is the negative log-likelihood. Note that the model is a 2-layer NN for the means and a 2-layer NN for the variances. See :ref:`ref_formulas` for the complete formulas.

    - :underline:`Shape of the output`: 4 parameters: 3 for the means and one for kappa.
    - :underline:`Deterministic tracking`: ?
    - :underline:`Probabilistic tracking`: We sample using rejection sampling defined in ( Directional Statistics (Mardia and Jupp, 1999)), implemented in `ref4 <https://github.com/jasonlaska/spherecluster>`_.

**FisherVonMisesMixtureDirectionGetter**: Not implemented yet.