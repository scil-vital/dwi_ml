.. _other_main_models:

Create your own model: Inherit from our models
==============================================

We have also prepared classes to help with common usages. Your model could easily inherit from them to benefit from what they have to offer. Each of them is a child of our main abstract model (as your own model should be, see


General models
--------------

``ModelWithPreviousDirections``
*******************************

Previous direction: you may need to format, at each position of the streamline, the previous direction. Use . It adds parameters for the previous direction and embedding choices.


``ModelWithDirectionGetter``
****************************

This is our model intented for tractography models (i.e. streamline generation models). It defines a layer of what we call the "directionGetter", which outputs a chosen direction for the next step of tractography propagation. It adds 2-layer neural network with the appropriate output size depending on the format of the output direction: 3 values (x, y, z) for regression, n values for classification on a sphere with N directions, etc. It also defines the compute_loss method, using an appropriate choice of loss in each case (ex, cosinus error for regression, negative log-likelihood for classification, etc.). For more information, see below:

For generative models, the ``get_tracking_directions`` method should be implemented to be used.


Models for usage with dwi inputs
--------------------------------

``ModelWithNeighborhood``
*************************

Neighborhood usage: the class adds parameters to deal with a few choices of neighborhood definitions.

``MainModelOneInput``
*********************

The abstract models makes no assumption of the type of data required. In this model here, we add the parameters necessary to add one input volume (ex: underlying dMRI data), choose this model, together with the DWIMLTrainerOneInput, and the volume will be interpolated and send to your model's forward method. Note that if you want to use many images as input, such as the FA, the T1, the dMRI, etc., this can still be considered as "one volume", if your prepare your hdf5 data accordingly by concatenating the images. We will see that again when explaining the hdf5.

``ModelOneInputWithEmbedding``
******************************

A sub-version from ``MainModelOneInput`` which also defines parameters to add an embedding layer.
