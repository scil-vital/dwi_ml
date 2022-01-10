6. Training your model
======================

Even tough training depends on your own model, most of the necessary code has been prepared here to deal with the data in the hdf5 file and create a batch sampler that can get streamlines and their associated inputs. All you need to do now is implement a model and its forward method.

The data from the hdf5 file created before will be loaded through the MultisubjectDataset. For more information on this, read page :ref:ref_data_containers.

In the please_copy_and_adapt folder, adapt the train_model.py script. Choose or implement a child version of the classes described below to fit your needs.

Batch sampler
-------------

These classes defines how to sample the streamlines available in the
MultiSubjectData.

**AbstractBatchSampler:**

- Defines the __iter__ method:

    - Finds a list of streamlines ids and associated subj that you can later load in your favorite way.

- Define the load_batch method:

    - Loads the streamlines associated to sampled ids. Can resample them.

    - Performs data augmentation (on-the-fly to avoid having to multiply data on disk) (ex: splitting, reversing, adding noise).

Child class : **BatchStreamlinesSamplerOneInput:**

- Redefines the load_batch method:

    - Now also loads the input data under each point of the streamline (and possibly its neighborhood), for one input volume.

You are encouraged to contribute to dwi_ml by adding any child class here.

Trainer
-------

**DWIMLAbstractTrainer:**

This class's main method is *train_and_validate()*:

- Creates DataLoaders from the data_loaders. Collate_fn will be the sampler.load_batch() method, and the dataset will be sampler.source_data.

- Trains each epoch by using compute_batch_loss, which should be implemented in each project's child class, on each batch. Saves the loss evolution and gradient norm in a log file.

- Validates each epoch (also by using compute_batch_loss on each batch, but skipping the backpropagation step). Saves the loss evolution in a log file.

After each epoch, a checkpoint is saved with current parameters. Training can be continued from a checkpoint using the script resume_training_from_checkpoint.py.

Visualizing training
--------------------

You can run "visualize_logs.py your_experiment" to see the evolution of the losses and gradient norm.

You can also use COMET to save results (code to be improved).

