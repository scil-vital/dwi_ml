.. _other_main_models:

Create your own model: Inherit from our models
==============================================

We have also prepared classes to help with common usages. Your model could easily inherit from them to benefit from what they have to offer. Each of them is a child of our main abstract model (as your own model should be, see


General models
--------------

``ModelWithPreviousDirections``
*******************************

In models, streamlines are often used as targets. But you may also need to use them as input. If your model iterates on streamline points and needs a fixed number of inputs at the time, you could use the N previous directions at each position of the streamline (see for instance in Poulin et al. 2017). This model adds parameters for the previous direction, plus embedding choices.


``ModelWithDirectionGetter``
****************************

This is our model intented for tractography models. It defines a layer of what we call the "directionGetter", which outputs a chosen direction for the next step of tractography propagation, in many possible formats, and knows how to compute the loss function accordingly. See the page :ref:`directionGetters` for more information.

It also contains a ``get_tracking_directions`` method, which should be implemented in your project to use this model for tractography.


Models for usage with DWI inputs
--------------------------------

``ModelWithNeighborhood``
*************************

Neighborhood usage: the class adds parameters to deal with a few choices of neighborhood definitions.

``ModelWithOneInput``
*********************

The ``MainAbstractModel`` makes no assumption of the type of data required. In this model here, we add the parameters necessary to add one input volume (ex: underlying dMRI data), choose this model, together with the DWIMLTrainerOneInput, and the volume will be interpolated and send to your model's forward method. Note that if you want to use many images as input, such as the FA, the T1, the dMRI, etc., this can still be considered as "one volume", if your prepare your hdf5 data accordingly by concatenating the images.

It defines parameters to add an embedding layer.
