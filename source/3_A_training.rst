3. Training your model
======================

Even tough training depends on your own model, we have prepared Trainers that can probably be used in any case.

3.1. Our trainers
-----------------

- They have a ``train_and_validate`` method that can be used to iterate on epochs (until a maximum number of iteration is reached, or a maximum number of bad epochs based on some loss).
- They save a checkpoint folder after each epoch, containing all information to resume the training any time.
- When a minimum loss value is reached, the model's parameters and states are save in a best_model folder.
- They save a good quantity of logs, both as numpy arrays (.npy logs) and online using Comet.ml.
- They know how to deal with the ``BatchSampler`` (which samples a list of streamlines to get for each batch) and with the ``BatchLoader`` (which gets data and performs data augmentation operations, if any).
- They prepare torch's optimizer (ex, Adam, SGD, RAdam), define the learning rate, etc.

3.2. Our Batch samplers and loaders
-----------------------------------

.. toctree::
    :maxdepth: 2

    3_B_MultisubjectDataset
    3_C_BatchSampler
    3_D_BatchLoader


3.3. Putting it all together
----------------------------

This class's main method is *train_and_validate()*:

- Creates torch DataLoaders from the data_loaders. Collate_fn will be the sampler.load_batch() method, and the dataset will be sampler.source_data.

- Trains each epoch by using compute_batch_loss, which should be implemented in each project's child class, on each batch. Saves the loss evolution and gradient norm in a log file.

- Validates each epoch (also by using compute_batch_loss on each batch, but skipping the backpropagation step). Saves the loss evolution in a log file.

After each epoch, a checkpoint is saved with current parameters. Training can be continued from a checkpoint using the script resume_training_from_checkpoint.py.

3.4. Visualizing logs
---------------------

You can run "visualize_logs.py your_experiment" to see the evolution of the losses and gradient norm.

You can also use COMET to save results (code to be improved).

3.5. Trainer with generation
----------------------------

toDO

