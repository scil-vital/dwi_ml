.. _main_abstract_model:

Create your own model: use MainModelAbstract
============================================

You are welcome to add your own project's model in dwi_ml!

Projects using diffusion imaging and tractography streamlines are usually quite heavy in memory. For this reason, dwi_ml is not only a space for models, but it also includes smart management of data loading during training (see :ref:`training` for more information). Our training objects and model objects are thus intertwined. For this reason, you should always make your model a child class of our **MainModelAbstract**.


The MainModelAbstract class
---------------------------

It is a derivate of torch's Module, to which we added methods to make it compatible with our Trainer. During training (or validation), our Trainer will perform these operations::

    for each epoch:
        batches = ... (batches are given by the BatchSampler)
        for each batch:
            data = ... (data is loaded and prepared by the BatchLoader)
            outputs = model(data)  # This calls model.forward()
            loss = model.compute_loss()

        model.save_checkpoint()

Our model:

- defines the way to save the model at each checkpoint [#f1]_ and once the training is finished, with predefined methods[#f2]_:

  - save_params_and_state
  - load_model_from_params_and_state

- defines the type of inputs the ``forward`` method will receive when called
  in the trainer.

- prepares a method ``compute_loss``, which will be called by our trainer
  during training / validation.

- has properties to tell BatchLoader how to prepare data, such as step_size, nb_points or compress_lines. These preprocessing steps are performed by the BatchLoader, but they probably influence strongly how the model performs, particularly in sequence-based models, as they change the length of streamlines. This is why these parameters have been added as main hyperparameters.

- has some internal values for easier management, such as ``self.device`` and
  ``self.context``.


.. [#f1] Checkpoint: after each epoch (which, in our field, can sometimes last hours or days!), our trainer saves on disk the current model and trainer states, to ensure that you can continue training if the training becomes stopped for any reason.

.. [#f2] Note that the ``forward`` and ``compute_loss`` methods are prepared but not implemented. You should implement them in your own project. See below.

Where to start?
---------------

As a first step, try to implement a child class of ``dwi.models.main_models.MainModelAbstract`` and see how you would implement the two following methods: ``forward()`` (the core of your model) and ``compute_loss()``.

Once you are more experimented, you can explore how to use methods already implemented in dwi_ml to improve your model, using, for instance, our models used for tractographie generation, or our models able to add a neighborhood to each input point, etc.

1. Create a new file in ``src/dwi_ml/models/projects`` named my_project.py

2. Start your project like this:

            .. image:: /_static/images/create_your_model.png
               :width: 500

3. Learn to use your model in our Trainer (see page :ref:`trainers`).

4. Before coding everything from scratch in our model, verify if it could inherit from our other models (see page :ref:`other_main_models`) to benefit from their methods.
