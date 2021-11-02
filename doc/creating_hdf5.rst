.. _ref_creating_hdf5:

4. Creating the hdf5 file
=========================

The possibility of laziness
***************************

We chose to base our code on the hdf5 data. One reason is that it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason for using hdf5 is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handling the data is called "lazy" in our project.

You will use the **create_hdf5_dataset.py** script to create a hdf5 file. You need to prepare config files to use this script (see lower).

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

**A note about data standardization**

Data is standardized (normalized) during data creation: data = (data - mean) / std. (Each features/modalities independently or not).

If all voxel were to be used, most of them would probably contain the background of the data, bringing the mean and std probably very close to 0. Thus, non-zero voxels only are used to compute the mean and std, or voxels inside the provided mask if any.

In the latest case, voxels outside the mask could have been set to NaN, but a test with the b0 as a mask showed that some streamlines had points outside the mask (probably due to data interpolation or to the skull-stripping technique of the b0 mask). The safer choice, chosen in dwi_ml, was to simply modify all voxels to data = (data - mean) / std, even voxels outside the mask.
