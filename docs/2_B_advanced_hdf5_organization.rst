.. _ref_creating_hdf5:

The hdf5 organization
=====================

Here is the output format created by dwiml_create_hdf5_dataset.py and recognized by our scripts:

.. code-block:: bash

    hdf5.attrs["version"] = the database version.
    hdf5.attrs['training_subjs'] = the list of str representing the training subjects.
    hdf5.attrs['validation_subjs'] = the list of str representing the validation subjects.
    hdf5.attrs['testing_subjs'] = the list of str representing the testing subjects.
    hdf5.attrs['means_and_stds'] = a dict with the (mean, std) for each volume group (if normalization across subjects is used), else 0, where std [=sqrt(variance)]. Each one is a vector of length = nb_features

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
    hdf5['subj1']['group1']['connectivity_label_volume'] = (the labels\' volume group) OR
    hdf5['subj1']['group1']['connectivity_nb_blocs'] = (a list of three integers)
    hdf5['subj1']['group1']['data_per_streamline'] = a HDF5 group of 2D numpy arrays

    # For volumes, other available data:
    hdf5['sub1']['group1']['affine']
    hdf5['sub1']['group1']['voxres']
    hdf5['sub1']['group1']['nb_features']

