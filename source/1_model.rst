1. Preparing your model
=======================

.. role:: underline
    :class: underline

The first task is to understand if your model can fit in our environment. Try to create your own model!


1.1. Main models
----------------

You should make your model a child class of our **MainModelAbstract**. It is a derivate of torch's Module, to which we added methods to load and save its parameters on disk. However, its `forward()` and `compute_loss()` methods are not implemented:

    .. image:: images/main_model_abstract.png
       :width: 600

    1. Create a new file in models.projects --> my_project.py
    2. Start like this:

    .. image:: images/create_your_model.png
       :width: 600

As you will discover when reading about our trainers, we have prepared them so that they will know which data they must send to your model's methods. You may change the variables `model.forward_uses_streamlines` and `model.loss_uses_streamlines` if you want the trainer to load and send the streamlines to your model. If your model also uses an input volume, see below, MainModelOneInput.

5.2. Other abstract models
--------------------------

We have also prepared child classes to help with common usages:

- Neighborhood usage: the class `ModelWithNeighborhood` adds parameters to deal with a few choices of neighborhood definitions.

- Previous direction: you may need to format, at each position of the streamline, the previous direction. Use `ModelWithPreviousDirections`. It adds parameters for the previous direction and embedding choices.

- MainModelOneInput: The abstract models makes no assumption of the type of data required. In this model here, we add the parameters necessary to add one input volume (ex: underlying dMRI data), choose this model, together with the DWIMLTrainerOneInput, and the volume will be interpolated and send to your model's forward method. Note that if you want to use many images as input, such as the FA, the T1, the dMRI, etc., this can still be considered as "one volume", if your prepare your hdf5 data accordingly by concatenating the images. We will see that again when explaining the hdf5.

    - ModelOneInputWithEmbedding: A sub-version also defined parameter to add an embedding layer.

- ModelWithDirectionGetter: This is our model intented for tractography models (i.e. streamline generation models). It defines a layer of what we call the "directionGetter", which outputs a chosen direction for the next step of tractography propagation. It adds 2-layer neural network with the appropriate output size depending on the format of the output direction: 3 values (x, y, z) for regression, n values for classification on a sphere with N directions, etc. It also defines the compute_loss method, using an appropriate choice of loss in each case (ex, cosinus error for regression, negative log-likelihood for classification, etc.). For more information, see :ref:`direction_getters`.


  For generative models, the `get_tracking_directions` should be implemented to be used.

  Then, see further how to track from your model using our Tracker.
