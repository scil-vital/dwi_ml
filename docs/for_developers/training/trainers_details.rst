
Trainers: the code explained
============================

THIS SECTION IS UNDER CONSTRUCTION.

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


Putting it all together
***********************

This class's main method is *train_and_validate()*:

- Creates torch DataLoaders from the data_loaders. Collate_fn will be the sampler.load_batch() method, and the dataset will be sampler.source_data.

- Trains each epoch by using compute_batch_loss, which should be implemented in each project's child class, on each batch. Saves the loss evolution and gradient norm in a log file.

- Validates each epoch (also by using compute_batch_loss on each batch, but skipping the backpropagation step). Saves the loss evolution in a log file.

After each epoch, a checkpoint is saved with current parameters. Training can be continued from a checkpoint using the script resume_training_from_checkpoint.py.

