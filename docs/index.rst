Welcome to DWI_ML documentation!
================================

This website is a guide to the github repository from the SCIL-VITAL organisation: https://github.com/scil-vital/dwi_ml/.

In this doc, we will present you everything included in this library for you to become either a developer or a user. Note that to get a full understanding of every line of code, you can browse further in each section.

Getting started
---------------

    .. toctree::
        :maxdepth: 1
        :titlesonly:
        :caption: Getting started

        getting_started

Explanations for users of pre-trained models
--------------------------------------------

Pages in this section explain how to use our scripts to use our pre-trained models.

- **Models**: If you want to use our pre-trained models, you may contact us for access to the models learned weights. They will be available online once publications are accepted.

- **Using hdf5**: In most cases, data must be organized correctly as a hdf5 before usage. See our page :ref:`ref_config_file` for an explanation.

    .. toctree::
        :maxdepth: 2
        :caption: Explanations for users (pre-trained)

        for_users/our_models
        for_users/hdf5
        for_users/tracking


Explanations for users of pre-coded models
------------------------------------------

Pages in this section are useful if you want to train a model based on pre-existing code, such as Learn2track or TractographyTransformers, using your favorite set of hyperparameters.


    .. toctree::
        :maxdepth: 2
        :caption: Explanations for users (re-train)

        for_users/from_start_to_finish


Explanations for developers
---------------------------

Page in this section explain more in details how the code is implemented in python.

- **Models**: The first aspect to explore are our models. Discover how you can create your model to fit with our structure. Many parent classes are available for you: if your model inherits from them, they will have access to everything each one offers. For instance, some models have instructions on how to receive inputs from MRI data, prepare inputs in a neighborhood, and use embedding. Other models have access to many options of loss functions for the context of tractography (cosine similarity, classification, Gaussian loss, Fisher von Mises, etc.).

- **Using hdf5**: Our library has been organized to use data in the hdf5 format. Our hdf5 data organization should probably be enough for your needs (see explanations on :ref:`ref_config_file`), but for more

- **Training a model**: Then, take a look at how we have implemented our trainers for an efficient management of heavy data.

- **Using your trained models**: Discover our objects allowing to perform a full tractography from a tractography-model.


    .. toctree::
        :maxdepth: 1
        :caption: Explanations for developers

        for_developers/models/index
        for_developers/hdf5/advanced_hdf5_organization
        for_developers/training/training
        for_developers/training/trainers_details
        for_developers/data_management/Advanced_data_containers
        for_developers/data_management/BatchSampler
        for_developers/data_management/BatchLoader
        for_developers/testing/general_testing
        for_developers/testing/tracking_objects