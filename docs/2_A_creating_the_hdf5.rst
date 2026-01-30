.. _ref_config_file:

2. Converting your data into a hdf5 file
========================================

2.1. The possibility of laziness
********************************

We chose to base our code on the hdf5 data. One reason is that it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handling the data is called "lazy" in our project.

The hdf5 may contain many groups of data. For instance, if your model needs an input volume and the streamlines as target, you might need one group for each. You might want to include tracking masks or any other required data.


2.2 How to organize your data?
******************************

This is how your data should be organized before trying to load your data as a hdf5 file. This structure should hold wether you work with hdf5 or BIDS. Below, we call "dwi_ml_ready" the folder with correct organization.

*Hint:* use symlinks to avoid doubling your data on disk!

**dwi_ml_ready**

This folder is the most important one and must be organized in a very precise way to be able to load the data as a hdf5 using our script **dwiml_create_hdf5_dataset**. Each subject should have the exact same sub-folders and files. Then, you can create a **config_file.json** that will tell the script what to include in the hdf5 file.

**Example:**

.. code-block:: bash

    {database_name}
    | dwi_ml_ready  =====> Each subject should contain the exact same sub-folders
                           and files, such as below. It is also possible to add
                           prefixes to the files (ex: subj1__t1.nii.gz) based on
                           the subject id.
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


2.3. Preparing the config file
******************************

To create the hdf5 file, you will need a config file such as below. HDF groups will be created accordingly for each subject in the hdf5.

.. code-block:: bash

    {
        "input": {
            "type": "volume",
            "files": ["dwi/dwi.nii.gz", "anat/t1.nii.gz", "dwi/*__dwi.nii.gz],
            "std_mask": [masks/some_mask.nii.gz]
        },
        "target": {
            "type": "streamlines",
            "files": ["tractograms/bundle1.trk", "tractograms/wholebrain.trk", "tractograms/*__wholebrain.trk"],
            "connectivity_matrix": "my_file.npy",
            "connectivity_nb_blocs": 6                    ( OR )
            "connectivity_labels": labels_volume_group,
            "dps_keys": ['dps1', 'dps2']
        }
        "bad_streamlines": {
            "type": "streamlines",
            "files": ["bad_tractograms/*"]
        }
        "wm_mask": {
            "type": "volume",
            "files": ["masks/wm_mask.nii.gz"]
        }
    }

|
Each group key will become the group's **name** in the hdf5. It can be anything you want. We suggest you keep it significative, ex 'input_volume', 'target_volume', 'target_streamlines'. In our scripts (ex, l2t_train_model.py, tt_train_model.py, etc), you will often be asked for the labels given to your groups.


Required attributes for each group
""""""""""""""""""""""""""""""""""

    - **"type"**: It must be recognized in dwi_ml. Currently, accepted datatype are:

        - 'volume': Volume groups will mimic nifti files. While creating the hdf5, you may concatenate many nifti files into a single group.

        - 'streamlines': Streamline groups will mimic tractogram files. Again, you may concatenate many .trk or .tck files in a single group, for instance you could concatenate many bundles per subject. Files must be inany format accepted by Dipy's *Stateful Tractogram*, such as .trk or .tck.

    - **"files"**: The files to concatenate into a single volume or a single tractogram. They must exist in every subject folder inside the root repository. That is: the files must be organized correctly on your computer (except if option 'enforce_files_presence is set to False).

        Note: There is the possibility to add a wildcard (\*), for instance if you files have variable prefixes (*_T1.nii.gz will include subj1_T1.nii.gz), or to include many files (bundles/*.trk will include all trk files in the bundles folder.).


Additional attributes for volume groups:
""""""""""""""""""""""""""""""""""""""""

    - **std_mask**: The name of the standardization mask (see Note 1). Data is standardized (normalized) during data creation: data = (data - mean_in_mask) / std_in_mask. If more than one files are given, the union (logical_or) of all masks is used (ex of usage: ["masks/wm_mask.nii.gz", "masks/gm_mask.nii.gz"] would use a mask of all the brain).

    - **"standardization"**: It defines the standardization (normalization) option applied to the volume group. It must be one of:

        - "all", to apply standardization to the final (concatenated) file, per subject.
        - "all_across_subjs", to apply standardization  to the final file, across all subjects
        - "independent", to apply it independently on the last dimension of the data; on each feature. For instance, for a fODF, this would apply standardization independently on each SH coefficient, per subject.
        - ("independent_across_subjs": not implemented!)
        - "per_file", to apply it independently on each file concatenated in the volume, per subject.
        - "per_file_across_subjs", to apply the same normalization to all subjects. See note 2.
        - "none", to skip this step (default)

****Note 1: why we use a mask for standardization**

If all voxel were to be used, most of them would probably contain the background of the data, bringing the mean and std probably very close to 0. Thus, non-zero voxels only are used to compute the mean and std, or voxels inside the provided mask if any. If a mask is provided, voxels outside the mask could have been set to NaN, but the safer choice made here was to simply modify all voxels [ data = (data - mean) / std ], even voxels outside the mask, with the mean and std of voxels in the mask.

****Note 2: how we apply standardization across subjects**

When we create the hdf5, to apply a the same standardization to all subjects, we could load volumes from all subjects in the training set at once, compute their mean and std. This could become heavy in memory, in data is big (typically 4D volumes) and if there are a lot of subjects. Rather, as we loop over all subjects to prepare the data, we use `Welford's algorithm <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance>`_ to compute the variance in an incremental way. The final mean and std [sqrt(variance)] are save as attributes of the hdf5. Models have access to this information in the hdf5 and can later standardize any new data it receives, even unseen data from the testing set.



Additional attributes for streamlines groups:
"""""""""""""""""""""""""""""""""""""""""""""

    - **connectivity_matrix**: The name of the connectivity matrix to associate to the streamline group. This matrix will probably be used as a mean of validation during training. Then, you also need to explain how the matrix was created, so that you can create the connectivity matrix of the streamlines being validated, in order to compare it with the expected result. ONE of the two next options must be given:

        - **connectivity_nb_blocs**: This explains that the connectivity matrix was created by dividing the volume space into regular blocs. See dwiml_compute_connectivity_matrix_from_blocs for a description. The value should be either an integers or a list of three integers.
        - **connectivity_labels**: This explains that the connectivity matrix was created by dividing the cortex into a list of regions associated with labels. The value must be the name of the associated labels file (typically a nifti file filled with integers).

    - **dps_keys**: List of data_per_streamline keys to keep in memory in the hdf5.



2.4. Creating the hdf5
**********************

You will use the **dwiml_create_hdf5_dataset** script to create a hdf5 file.

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

You may later investigate the organization of your hdf5 with the script **dwiml_hdf5_print_architecture**.

.. toctree::
    :maxdepth: 1
    :caption: Detailed explanations for developers:

    2_B_advanced_hdf5_organization
