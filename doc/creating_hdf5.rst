.. _ref_creating_hdf5:

Creating the hdf5 file
======================

About hdf5
**********

We chose to base on code on the hdf5 data. First, it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason for using hdf5 is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handeling the data is called "lazy" in our project.

You will use the create_hdf5_dataset.py script to create a hdf5 file. You need to prepare config files to use this script (see lower). This is the first step found in please_copy_and_adapt/run_project.sh.

Config file
***********

Expected json config for the groups in your hdf5:

.. code-block:: bash

    {
        "group1": {
            "type": "volume",
            "files": ["dwi/dwi_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz"]
             },
        "group2": {
            ...
             }
    }

The group names could be 'input_volume', 'target_volume', 'target_directions', or anything. Make sure your training scripts and your model'sbatch_sampler use the same keys. The groups 'files' must exist in every subject folder inside dwi_ml_ready. The groups 'type' must be recognized in dwi_ml. Currently, accepted datatype is:

    - 'volume': for instance, a dwi, an anat, mask, t1, fa, etc.

Currently, only one group of streamlines is accepted for your data, and thus the streamlines are not integreated in the config file but are considered separately. During the data creation, see the option --bundles.

Creating the hdf5
*****************

.. code-block:: bash

    create_hdf5_dataset.py --force --name $name --std_mask $mask_for_standardization \
            --bundle "my_bundles" --space $space $database_folder $config_file \
            $training_subjs $validation_subjs --enforce_bundles_presence True

Final hdf5 file structure
*************************

If you would rather create your hdf5 file with your own script, here is the output format created by create_hdf5_dataset.py and recognized by the multi_subject_containers:

.. code-block:: bash

    hdf5.attrs["version"] = the database version.
    hdf5.attrs['training_subjs'] = the list of str representing the training subjects.

    hdf5.keys() are the subjects.
    hdf5['subj1'].keys() are the groups from the config_file + 'streamlines'.
    hdf5['subj1']['group1'].attrs['type'] = 'volume' or 'streamlines'.
    hdf5['subj1']['group1']['data'] is the data.

    For streamlines, other available data:
    hdf5['subj1']['streamlines']['data']
    hdf5['subj1']['streamlines']['offsets']
    hdf5['subj1']['streamlines']['lengths']
    hdf5['subj1']['streamlines']['euclidean_lengths']