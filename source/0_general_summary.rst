General summary
===============

Our repository includes:

    - **Models**: You will discover how you can create your model to fit with our structure. As a first step, try to implement a child class of `dwi.models.main_models.MainModelAbstract` and see how you would implement the two following methods: `forward()` (the core of your model) and `compute_loss()`.

      Once you are more experimented, you can explore how to use methods already implemented in dwi_ml to improve your model, using, for instance, our models used for tractographie generation, or our models able to add a neighborhood to each input point, etc.

    - **Training a model**: If your model fits well with our structures, you can use our Trainer, wich itself uses a MultiSubjectDataset (it knows how to get data in the hdf5), a batch sampler (it knows how to sample a list of chosen streamlines for a batch) and a batch loader (it knows how to load the data using the MultiSubjectDataset, and modify the streamlines based on your model's requirements, for instance, adding noise or compressing / changing the step size / reversing / splitting the streamlines).

      The trainer also saves the state of the model and of the optimizer in a checkpoint directory, to allow resuming your experiment if it was stopped prematurely.

      It might be difficult to get used to it. The easier is probably to copy an existing script and start from there!

    - **How to see logs**: The trainer save a lot of information at each epoch: the training or validation loss in particular. It can also send the information to comet on the fly; go discover their incredible website.

    - **Tractography models**: Many more options are available for tractography models. You will discover them throughout the following pages.


.. toctree::
    :maxdepth: 3
    :caption: Detailed table of content:

    1_A_model
    2_A_creating_the_hdf5.rst
    3_A_training
    4_tracking