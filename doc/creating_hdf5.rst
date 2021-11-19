.. _ref_creating_hdf5:

4. Creating the hdf5 file
=========================

The possibility of laziness
***************************

We chose to base our code on the hdf5 data. One reason is that it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason for using hdf5 is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handling the data is called "lazy" in our project.

You will use the **create_hdf5_dataset.py** script to create a hdf5 file. You need to prepare config files to use this script (see :ref:`ref_config_file`).

Creating the hdf5
*****************

Exemple of use: (See please_copy_and_adapt/ALL_STEPS.sh) for a more thorough example).

.. code-block:: bash

    create_hdf5_dataset.py --force --name $name --std_mask $mask --space $space \
            --enforce_bundles_presence True \
            $database_folder/dwi_ml_ready $database_folder/hdf5 $config_file \
            $training_subjs $validation_subjs

Here is the output format created by create_hdf5_dataset.py and recognized by the multi_subject_containers:

.. code-block:: bash

    hdf5.attrs["version"] = the database version.
    hdf5.attrs['training_subjs'] = the list of str representing the training subjects.
    hdf5.attrs['validation_subjs'] = the list of str representing the validation subjects.
    hdf5.attrs['testing_subjs'] = the list of str representing the testing subjects.

    hdf5.keys() are the subjects.
    hdf5['subj1'].keys() are the groups from the config_file.
    hdf5['subj1']['group1'].attrs['type'] = 'volume' or 'streamlines'.
    hdf5['subj1']['group1']['data'] is the data.

    For streamlines, other available data:
    hdf5['subj1']['group1']['offsets']
    hdf5['subj1']['group1']['lengths']
    hdf5['subj1']['group1']['euclidean_lengths']
