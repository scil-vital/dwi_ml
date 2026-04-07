Welcome to DWI_ML documentation!
================================

This website is a guide to the github repository from the SCIL-VITAL organisation: https://github.com/scil-vital/dwi_ml/. DWI_ML is a toolkit for Diffusion Magnetic Resonance Imaging (dMRI) analysis
using machine learning and deep learning methods. It is mostly focused on the tractography derivatives of dMRI.


        .. image:: /_static/images/logo_dwi_ml_emma_avec_texte.png
            :align: center
            :width: 500

In this doc, we will present you everything included in this library for you to become either a developer or a user.

On this page:

    - :ref:`section_install`
    - :ref:`section_users`
    - :ref:`section_advanced_users`
    - :ref:`section_developers`

.. _section_install:

1. Installing dwi_ml
--------------------

    .. toctree::
        :maxdepth: 1
        :titlesonly:
        :caption: Installing dwi_ml

        getting_started

.. _section_users:

2. Explanations for users of pre-trained models (Learn2track, Transformers)
---------------------------------------------------------------------------

Pages in this section explain how to use our scripts to use our pre-trained models.

- **1. Downloading models**: If you want to use our pre-trained models, you may contact us for access to the models learned weights. They will be available online once publications are accepted.

- **2. Organizing your data**: In most cases, data must be organized correctly as a hdf5 before usage. Follow the link below for an explanation.

    - :ref:`hdf5_usage`

- **3. Using our models to perform tractography**: Use our models to track on your own subjects!

    - :ref:`tractography_models`

- **OR, Using our models to denoise your tractograms**: (upcoming)

    - :ref:`denoising_models`

.. --------------------Hidden toctree: ---------------


.. toctree::
    :maxdepth: 2
    :hidden:
    :caption: Explanations for users (pre-trained)

    for_users/hdf5
    for_users/models/tractography_models
    for_users/models/denoising_models

.. _section_advanced_users:

3. Explanations for advanced users: train a model with your own hyperparameters
-------------------------------------------------------------------------------

Pages in this section are useful if you want to train a model based on pre-existing code, such as Learn2track or TractographyTransformers, using your favorite set of hyperparameters.


    .. toctree::
        :maxdepth: 2
        :caption: Explanations for explorers

        for_users/from_start_to_finish_tracking
        for_users/from_start_to_finish_denoising
        for_users/visu_logs

.. _section_developers:

4. Explanations for developers: create your own model
-----------------------------------------------------

Page in this section explain more in details how the code is implemented in python.

- **1. Create your models**: The first aspect to explore are our models. Discover how you can create your model to fit with our structure. Many parent classes are available for you: if your model inherits from them, they will have access to everything each one offers. For instance, some models have instructions on how to receive inputs from MRI data, prepare inputs in a neighborhood, and use embedding. Other models have access to many options of loss functions for the context of tractography (cosine similarity, classification, Gaussian loss, Fisher von Mises, etc.).

    - :ref:`create_your_model`

        - :ref:`main_abstract_model`
        - :ref:`other_main_models`
        - :ref:`direction_getters`

- **2. Explore our hdf5 organization**: Our library has been organized to use data in the hdf5 format. Our hdf5 data organization should probably be enough for your needs.

    - :ref:`hdf5_usage`
    - :ref:`creating_hdf5`

- **3. Train your model**: Take a look at how we have implemented our trainers for an efficient management of heavy data. Note that our trainer uses Data Management classes such as our BathLoader and BatchSampler. See below for more information.

    - :ref:`trainers`
    - :ref:`data_management_index`

        - :ref:`ref_data_containers`
        - :ref:`batch_sampler`
        - :ref:`batch_loaders`

- **4. Use your trained model**: This step depends on your model. For tractography models, discover our objects allowing to perform a full tractography from a tractography model. You can also see our pages for Learntrack and TractoTransformer usage: :ref:`tractography_models`.

    - :ref:`tracking`


.. --------------------Hidden toctree: For developers---------------

.. toctree::
    :maxdepth: 3
    :caption: Explanations for developers
    :hidden:

    for_developers/models/index
    for_developers/hdf5/advanced_hdf5_organization
    for_developers/training/training
    for_developers/data_management/index
    for_developers/testing/tracking_objects


.. --------------------Hidden toctree: scripts ---------------

.. toctree::
    :maxdepth: 2
    :caption: Scripts (--help)
    :hidden:

    automatic_doc/index_automatic.rst
