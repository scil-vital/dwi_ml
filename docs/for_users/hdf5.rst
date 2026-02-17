.. _ref_config_file:

Converting your data into a hdf5 file
=====================================

The possibility of laziness
***************************

We chose to base our code on the hdf5 data. One reason is that it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handling the data is called "lazy" in our project.

The hdf5 may contain many groups of data. For instance, if your model needs an input volume and the streamlines as target, you might need one group for each. You might want to include tracking masks or any other required data.

Volume groups will mimic nifti files. While creating the hdf5, you may concatenate many nifti files into a single group.

Streamline groups will mimic tractogram files. Again, you may concatenate many .trk or .tck files in a single group, for instance you could concatenate many bundles per subject.


How to organize your data?
**************************

We suggest that your data should be organized correctly on your computer, such as described below.

This is how your data should be organized before trying to load your data as a hdf5 file. This structure should hold wether you work with hdf5 or BIDS. Below, we call "dwi_ml_ready" the folder with correct organization.

*Hint:* use symlinks to avoid doubling your data on disk!

**dwi_ml_ready**

This folder is the most important one and must be organized in a very precise way to be able to load the data as a hdf5 using our script **create_hdf5_dataset.py**. Each subject should have the exact same sub-folders and files. Then, you can create a **config_file.json** that will tell the script what to include in the hdf5 file.

**Example:**

.. code-block:: bash

    {database_name}
    | original  =====> Organized as you wish but if you intend on using
                       tractoflow, you should organize it as below.
        | {subject_id}
            | dwi.nii.gz
            | bval
            | bvec
            | t1.nii.gz
    | preprocessed =====>  Organized as you wish.
        | {subject_id}
            | Ex: Tractoflow folders
            | Ex: bundles from Recobundles
    | dwi_ml_ready  =====> Each subject should contain the exact same sub-folders
                           and files, such as below. It is also possible to add
                           prefixes to the files (ex: subj1__t1.nii.gz) based on
                           the subject id. For instance:
        | {subject_id}
            | anat
                | t1.nii.gz
                | wm_map.nii.gz
            | dwi
                | dwi_preprocessed.nii.gz
                | bval_preprocessed
                | bvec_preprocessed
                | fa.nii.gz
            | bundles
                | {bundle1}.tck
            | masks
                | wm.nii.gz
        | ...


Preparing the config file
*************************

To create the hdf5 file, you will need a config file such as below. HDF groups will be created accordingly for each subject in the hdf5.

.. code-block:: bash

    {
        "input": {
            "type": "volume",
            "files": ["dwi/dwi.nii.gz", "anat/t1.nii.gz", "dwi/*__dwi.nii.gz], --> Will get, for instance, subX__dwi.nii.gz
            "standardization": "all",
            "std_mask": [masks/some_mask.nii.gz]
             },
        "target": {
            "type": "streamlines",
            "files": ["tractograms/bundle1.trk", "tractograms/wholebrain.trk", "tractograms/*__wholebrain.trk"], ----> Will get, for instance, sub1000__bundle1.trk
            "connectivity_matrix": "my_file.npy",
            "connectivity_nb_blocs": 6  ---> OR
            "connectivity_labels": labels_volume_group,
            "dps_keys": ['dps1', 'dps2']
             }
        "bad_streamlines": {
            "type": "streamlines",
            "files": ["bad_tractograms/*"] ---> Will get all trk and tck files.
             }
        "wm_mask": {
            "type": "volume",
            "files": ["masks/wm_mask.nii.gz"]
            }
    }

|

General group attributes in the config file:
""""""""""""""""""""""""""""""""""""""""""""

Each group key will become the group's **name** in the hdf5. It can be anything you want. We suggest you keep it significative, ex 'input_volume', 'target_volume', 'target_directions'. In other scripts (ex, l2t_train_model.py, tt_train_model.py, etc), you will often be asked for the labels given to your groups.

Each group may have a number of parameters:

    - **"type"**: It must be recognized in dwi_ml. Currently, accepted datatype are:

        - 'volume': for instance, a dwi, an anat, mask, t1, fa, etc.
        - 'streamlines': for instance, a .trk, .tck file (any format accepted by Dipy's *Stateful Tractogram*).

    - **"files"**: The listed file(s) must exist in every subject folder inside the root repository. That is: the files must be organized correctly on your computer (except if option 'enforce_files_presence is set to False). If there are more than one files, they will be concatenated (on the 4th dimension for volumes, using the union of tractograms for streamlines).

        - There is the possibility to add a wildcard (\*).

Additional attributes for volume groups:
""""""""""""""""""""""""""""""""""""""""

    - **std_mask**: The name of the standardization mask. Data is standardized (normalized) during data creation: data = (data - mean_in_mask) / std_in_mask. If more than one files are given, the union (logical_or) of all masks is used (ex of usage: ["masks/wm_mask.nii.gz", "masks/gm_mask.nii.gz"] would use a mask of all the brain).

    - **"standardization"**: It defined the standardization option applied to the volume group. It must be one of:

        - "all", to apply standardization (normalization) to the final (concatenated) file.
        - "independent", to apply it independently on the last dimension of the data (ex, for a fODF, it would apply it independently on each SH).
        - "per_file", to apply it independently on each file included in the group.
        - "none", to skip this step (default)

****A note about data standardization**

If all voxel were to be used, most of them would probably contain the background of the data, bringing the mean and std probably very close to 0. Thus, non-zero voxels only are used to compute the mean and std, or voxels inside the provided mask if any. If a mask is provided, voxels outside the mask could have been set to NaN, but the simpler choice made here was to simply modify all voxels [ data = (data - mean) / std ], even voxels outside the mask, with the mean and std of voxels in the mask. Mask name is provided through the config file. It is formatted as a list: if many files are listed, the union of the binary masks will be used.


Additional attributes for streamlines groups:
"""""""""""""""""""""""""""""""""""""""""""""

    - **connectivity_matrix**: The name of the connectivity matrix to associate to the streamline group. This matrix will probably be used as a mean of validation during training. Then, you also need to explain how the matrix was created, so that you can create the connectivity matrix of the streamlines being validated, in order to compare it with the expected result. ONE of the two next options must be given:

        - **connectivity_nb_blocs**: This explains that the connectivity matrix was created by dividing the volume space into regular blocs. See dwiml_compute_connectivity_matrix_from_blocs for a description. The value should be either an integers or a list of three integers.
        - **connectivity_labels**: This explains that the connectivity matrix was created by dividing the cortex into a list of regions associated with labels. The value must be the name of the associated labels file (typically a nifti file filled with integers).

    - **dps_keys**: List of data_per_streamline keys to keep in memory in the hdf5.

Creating the hdf5
******************

You will use the **dwiml_create_hdf5_dataset.py** script to create a hdf5 file.

.. code-block:: bash

    dwi_ml_folder=YOUR_PATH
    hdf5_file=YOUR_OUT_FILE.hdf5
    config_file=YOUR_FILE.json
    training_subjs=YOUR_FILE.txt
    validation_subjs=YOUR_FILE.txt
    testing_subjs=YOUR_FILE.txt

    dwiml_create_hdf5_dataset.py --name $name --std_mask $mask --space $space \
            --enforce_files_presence True \
            $dwi_ml_folder $hdf5_file $config_file \
            $training_subjs $validation_subjs $testing_subjs
