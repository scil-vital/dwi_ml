.. _trainers:

Training your model
===================

If your model fits well with our structures, you can use our Trainer. If your model does not have specific needs, our Trainer should already be sufficient for you, and you can read section Using our Trainer below. Else, if you need to modify something, we explain our class more in detail below.

The trainer:

- Runs training and validation for all batchs, for a chosen number of epochs.
- Saves the state of the model and of the optimizer in a checkpoint directory, to allow resuming your experiment if it was stopped prematurely.


1. Our choices of trainers
--------------------------

``DWIMLTrainer``
************************

This is the main class. For every batch, it loads the chosen streamlines and uses the model, as explained in section 2 below.

``DWIMLTrainerOneInput``
************************

This trainer additionally loads one volume group and accessed the coordinates at each point of your streamlines, or possibly in a neighborhood at each coordinate. Of note, this is done as a separate step, and not through torch's DataLoaders (see explanation in :ref:`batch_loaders`), because interpolation of data is faster through GPU, if you have access, but DataLoaders always work on CPU.

This trainer is expected to be used with a child of ``ModelWithOneInput`` (see page :ref:`other_main_models`).

``DWIMLTrainerOneInputWithGVPhase``
***********************************

We will soon publish how we have used a new generation-validation phase to supervise our models.


2. Using a trainer for your model
---------------------------------

This is an example of basic script that you could create to train your model with our trainer. It will require:

- Your model
- An instance of our object ``MultiSubjectDataset``: it knows how to get data in the hdf5, possibly in a lazy way. See :ref:`ref_data_containers` for more information.
- An instance of a ``BatchSampler``: it knows how to sample a list of chosen streamlines for a batch. See :ref:`batch_sampler` for more information.
- An instance of a ``BatchLoader``: it knows how to load the data using the ``MultiSubjectDataset``, and how to modify the streamlines based on your model's requirements, for instance, adding noise or compressing / changing the step size / reversing / splitting the streamlines. See :ref:`batch_loader` for more information.

Your final python script could look like::

    # Loading the data, possibly with lazy option
    dataset = MultiSubjectDataset(hdf5_file)
    dataset.load_data()

    # Preparing your model
    model = myModel(args)

    # Preparing the BatchSampler
    batch_sampler = DWIMLBatchIDSampler(
            dataset=dataset, streamline_group_name=streamline_group_name)

    # Preparing the BatchLoader.
    batch_loader = DWIMLBatchLoaderOneInput(
            dataset=dataset, model=model,
            input_group_name=input_group_name,
            streamline_group_name=streamline_group_name)

    # Preparing your trainer
    trainer = DWIMLTrainerOneInput(
            model=model, experiments_path=experiments_path,
            experiment_name=experiment_name, batch_sampler=batch_sampler,
            batch_loader=batch_loader)

    # Run the training!
    trainer.train_and_validate()




3. Visualizing logs
---------------------

The trainer save a lot of information at each epoch: the training or validation loss in particular. It can send the information to Comet.ml on the fly; go discover their incredible website. Alternatively, you can run ``visualize_logs your_experiment`` to see the evolution of the losses and gradient norm.

Example of Comet.ml view:

        .. image:: /_static/images/example_comet.png
            :width: 1500