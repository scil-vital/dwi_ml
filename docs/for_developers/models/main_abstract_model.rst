.. _main_abstract_model:

Create your own model
=====================

You should always make your model a child class of our **MainModelAbstract**. It is a derivate of torch's Module, to which we added methods to load and save its parameters on disk during checkpoints [#f1]_. Its ``forward()`` and ``compute_loss()`` methods are prepared but not implemented:

            .. image:: /_static/images/main_model_abstract.png
               :width: 400

As you will discover when reading about our trainers, we have prepared them so that they will know which data they must send to your model's methods. During training, the trainer calls these two methods:

.... ADD PICTURE


How to proceed
--------------

As a first step, try to implement a child class of `dwi.models.main_models.MainModelAbstract` and see how you would implement the two following methods: `forward()` (the core of your model) and `compute_loss()`.

Once you are more experimented, you can explore how to use methods already implemented in dwi_ml to improve your model, using, for instance, our models used for tractographie generation, or our models able to add a neighborhood to each input point, etc.

1. Create a new file in ``models.projects`` named my_project.py
2. Start your project like this:

            .. image:: /_static/images/create_your_model.png
               :width: 500

3. Learn to use your model in our Trainer (see page :ref:`trainers`).

4. Before coding everything from scratch in our model, verify if it could inherit from our other models (see page :ref:`other_main_models`) to benefit from their methods.





----------------------

.. [#f1] Checkpoint: after each epoch, our trainer saves on disk the current model and trainer state, to ensure that you can continue training if the training becomes stopped for any reason.