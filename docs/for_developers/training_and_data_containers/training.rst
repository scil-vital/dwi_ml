.. _trainers:

Training your model
===================

If your model fits well with our structures, you can use our Trainer, wich itself uses a MultiSubjectDataset (it knows how to get data in the hdf5), a batch sampler (it knows how to sample a list of chosen streamlines for a batch) and a batch loader (it knows how to load the data using the MultiSubjectDataset, and modify the streamlines based on your model's requirements, for instance, adding noise or compressing / changing the step size / reversing / splitting the streamlines).

      The trainer also saves the state of the model and of the optimizer in a checkpoint directory, to allow resuming your experiment if it was stopped prematurely.

      It might be difficult to get used to it. The easier is probably to copy an existing script and start from there!

Even tough training depends on your own model, we have prepared Trainers that can probably be used in any case.

Our trainers
------------

- They have a ``train_and_validate`` method that can be used to iterate on epochs (until a maximum number of iteration is reached, or a maximum number of bad epochs based on some loss).
- They save a checkpoint folder after each epoch, containing all information to resume the training any time.
- When a minimum loss value is reached, the model's parameters and states are save in a best_model folder.
- They save a good quantity of logs, both as numpy arrays (.npy logs) and online using Comet.ml.
- They know how to deal with the ``BatchSampler`` (which samples a list of streamlines to get for each batch) and with the ``BatchLoader`` (which gets data and performs data augmentation operations, if any).
- They prepare torch's optimizer (ex, Adam, SGD, RAdam), define the learning rate, etc.

The ``train_and_validate``'s action, in short, is:

.. code-block:: python

    for epoch in range(nb_epochs):
        set_the_learning_rate
        self.train_one_epoch()
        self.validate_one_epoch()
        if this_is_the_best_epoch:
            save_best_model
        save_checkpoint

Where ``train_one_epoch`` does:

.. code-block:: python

    for batch in batches:
        self.run_one_batch()
        self.back_propagation()

And ``validate_one_epoch`` runs the batch but does not do the back-propagation.

Finally, ``run_one_batch`` is not implemented in the ``DWIMLAbstractTrainer`` class, as it depends on your model.

DWIMLTrainerOneInput
--------------------

So far, we have prepared one child Trainer class, which loads the streamlines and one volume group. It can be used with the MainModelOneInput, as described earlier. This class is used by Learn2track and by TractographyTransformers; you can rely on them to discover how to use it.


Our Batch samplers and loaders
-----------------------------------

Putting it all together
----------------------------

This class's main method is *train_and_validate()*:

- Creates torch DataLoaders from the data_loaders. Collate_fn will be the sampler.load_batch() method, and the dataset will be sampler.source_data.

- Trains each epoch by using compute_batch_loss, which should be implemented in each project's child class, on each batch. Saves the loss evolution and gradient norm in a log file.

- Validates each epoch (also by using compute_batch_loss on each batch, but skipping the backpropagation step). Saves the loss evolution in a log file.

After each epoch, a checkpoint is saved with current parameters. Training can be continued from a checkpoint using the script resume_training_from_checkpoint.py.

Visualizing logs
---------------------

You can run "visualize_logs.py your_experiment" to see the evolution of the losses and gradient norm.

You can also use COMET to save results (code to be improved).

Trainer with generation
----------------------------

toDO



    - **How to see logs**: The trainer save a lot of information at each epoch: the training or validation loss in particular. It can also send the information to comet on the fly; go discover their incredible website.


General testing of a model
--------------------------

This step depends on your model and your choice of metrics, but in generative models, you probably want to track on new data and verify the quality of your reconstruction. We have prepared a script that allows you to track from a model.


2. The trainer abstract :

    - Uses a model-dependent child implementation of the **BatchSamplerAbstract**:

        - For instance, the BatchSequenceSampler creates batches of streamline ids that will be used for each batch iteratively through *__iter__*. Those chosen streamlines can then be loaded and processed with *load_batch*, which also uses data augmentation.

        - For instance, the BatchSequencesSamplerOneInputVolume then uses the generated streamlines to load one underlying input for each timestep of each streamline.

        - The BatchSampler uses the **MultiSubjectDataset** (or lazy)

            - Creates a list of subjects. Using *self.load_data()*, it loops on all subjects (for either the training set or the validation set) and loads the data from the hdf5 (lazily or not).

            - The list is a **DataListForTorch** (or lazy). It contains the subjects but also common parameters such as the feature sizes of each input.

            - The elements are **SubjectData** (or lazy)

                - They contain both the volumes and the streamlines, with the subject ID.

                - Volumes are **MRIData** (or lazy). They contain the data and affine.

                - Streamlines are **SFTData** (or lazy). They contain all information necessary to create a Stateful Tractogram from the streamlines. In the lazy version, streamlines are not loaded all together but read when needed from the **LazyStreamlinesGetter**.
