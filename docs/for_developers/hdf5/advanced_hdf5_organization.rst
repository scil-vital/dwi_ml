.. _ref_creating_hdf5:

The hdf5 structure
==================

The current hdf5 organization probably suffises for your needs. See page :ref:`ref_config_file` for usage explanation.

Here is the output format created by dwiml_create_hdf5_dataset.py and recognized by our scripts. It can also be investigated by running script ``dwiml_hdf5_print_architecture`` with an existing hdf5.

.. code-block:: bash

    hdf5.attrs["version"] = the database version.
    hdf5.attrs['training_subjs'] = the list of str representing the training subjects.
    hdf5.attrs['validation_subjs'] = the list of str representing the validation subjects.
    hdf5.attrs['testing_subjs'] = the list of str representing the testing subjects.

    # hdf5.keys() are the subjects.
    hdf5['subj1'].keys() are the groups from the config_file.
    hdf5['subj1']['group1'].attrs['type'] = 'volume' or 'streamlines'.
    hdf5['subj1']['group1']['data'] is the data.

    # For streamlines, other available data:
    # (from the data:)
    hdf5['subj1']['group1']['offsets']
    hdf5['subj1']['group1']['lengths']
    hdf5['subj1']['group1']['euclidean_lengths']
    # (from the space attributes:)
    hdf5['subj1']['group1']['space']
    hdf5['subj1']['group1']['affine']
    hdf5['subj1']['group1']['dimensions']
    hdf5['subj1']['group1']['voxel_sizes']
    hdf5['subj1']['group1']['voxel_order']
    # (others:)
    hdf5['subj1']['group1']['connectivity_matrix']
    hdf5['subj1']['group1']['connectivity_matrix_type'] = 'from_blocs' or 'from_labels'
    hdf5['subj1']['group1']['connectivity_label_volume'] (the labels\' volume group) OR
    hdf5['subj1']['group1']['connectivity_nb_blocs'] (a list of three integers)
    hdf5['subj1']['group1']['data_per_streamline'] (a HDF5 group of 2D numpy arrays)

    # For volumes, other available data:
    hdf5['sub1']['group1']['affine']
    hdf5['sub1']['group1']['voxres']
    hdf5['sub1']['group1']['nb_features']

If this is not enough for you, you may investigate our file ``dwi_ml/data/hdf5/hdf5_creation.py``.