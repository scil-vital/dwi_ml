.. _ref_config_file:

2. Converting your data into a hdf5 file
========================================

2.1. The possibility of laziness
********************************

We chose to base our code on the hdf5 data. One reason is that it allows to regroup your data in an organized way to ensure that all you data is present. But the main reason is that it is then possible to load only some chosen streamlines for each batch in the training set instead of having to keep all the streamlines in memory, which can be very heavy. This way of handling the data is called "lazy" in our project.

The hdf5 may contain many groups of data. For instance, if your model needs an input volume and the streamlines as target, you might need one group for each. You might want to include tracking masks or any other required data.

Volume groups will mimic nifti files. While creating the hdf5, you may concatenate many nifti files into a single group.

Streamline groups will mimic tractogram files. Again, you may concatenate many .trk or .tck files in a single group, for instance you could concatenate many bundles per subject.


2.2 How to organize your data?
******************************

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


2.3. Preparing the config file
******************************

To create the hdf5 file, you will need a config file such as below. HDF groups will be created accordingly for each subject in the hdf5.

.. code-block:: bash

    {
        "input": {
            "type": "volume",
            "files": ["dwi/dwi.nii.gz", "anat/t1.nii.gz", "dwi/*__dwi.nii.gz], --> Will get, for instance, subX__dwi.nii.gz
            "standardization": "all"
             },
        "target": {
            "type": "streamlines",
            "files": ["tractograms/bundle1.trk", "tractograms/wholebrain.trk", "tractograms/*__wholebrain.trk"] ----> Will get, for instance, sub1000__bundle1.trk
             }
        "bad_streamlines": {
            "type": "streamlines",
            "files": ["bad_tractograms/ALL"] ---> Will get all trk and tck files.
             }
        "wm_mask": {
            "type": "volume",
            "files": ["masks/wm_mask.nii.gz"]
            }
    }

General group attributes in the config file:

- The group's **name** could be 'input_volume', 'target_volume', 'target_directions', or anything.

    - We will see further how to tell your model and your batch loader the group names of interest.

- The group's **"files"** must exist in every subject folder inside a repository. That is: the files must be organized correctly on your computer. See (except if option 'enforce_files_presence is set to False).

    - There is the possibility to add a wildcard (*) that will be replaced by the subject's id while loading. Ex: anat/\*__t1.nii.gz would become anat/subjX__t1.nii.gz.
    - For streamlines, there is the possibility to use 'ALL' to load all tractograms present in a folder.
    - The files from each group will be concatenated in the hdf5 (either as a final volume or as a final tractogram).

- The groups **"type"** must be recognized in dwi_ml. Currently, accepted datatype are:

    - 'volume': for instance, a dwi, an anat, mask, t1, fa, etc.
    - 'streamlines': for instance, a .trk, .tck file (anything accepted by Dipy's Stateful Tractogram).

Additional attribute for volume groups:

- The groups **"standardization"** must be one of:

    - "all", to apply standardization (normalization) to the final (concatenated) file
    - "independent", to apply it independently on the last dimension of the data (ex, for a fODF, it would apply it independently on each SH).
    - "per_file", to apply it independently on each file included in the group.
    - "none", to skip this step (ex: for binary masks, which must stay binary).

****A note about data standardization**

    Data is standardized (normalized) during data creation: data = (data - mean) / std.

    If all voxel were to be used, most of them would probably contain the background of the data, bringing the mean and std probably very close to 0. Thus, non-zero voxels only are used to compute the mean and std, or voxels inside the provided mask if any. If a mask is provided, voxels outside the mask could have been set to NaN, but the simpler choice made here was to simply modify all voxels [ data = (data - mean) / std ], even voxels outside the mask, with the mean and std of voxels in the mask. Mask name for each subject is provided using --std_mask in the script create_hdf5_dataset.py.

2.4. Creating the hdf5
**********************

You will use the **create_hdf5_dataset.py** script to create a hdf5 file. You need to prepare config files to use this script (see :ref:`ref_config_file`).

Exemple of use: (See also please_copy_and_adapt/ALL_STEPS.sh)

.. code-block:: bash

    dwi_ml_folder=YOUR_PATH
    hdf5_folder=YOUR_PATH
    config_file=YOUR_FILE.json
    training_subjs=YOUR_FILE.txt
    validation_subjs=YOUR_FILE.txt
    testing_subjs=YOUR_FILE.txt

    dwiml_create_hdf5_dataset.py --name $name --std_mask $mask --space $space \
            --enforce_files_presence True \
            $dwi_ml_folder $hdf5_folder $config_file \
            $training_subjs $validation_subjs $testing_subjs

P.S How to get data?
********************

Here in the SCIL lab, we often suggest to use the Tractoflow pipeline to process your data. If you need help for the pre-processing and reorgnization of your database, consult the following pages:

.. toctree::
    :maxdepth: 2

    2_B_preprocessing


Organizing data from tractoflow to dwi_ml_ready
-----------------------------------------------

If you used tractoflow to preprocess your data, you may organize automatically the dwi_ml_ready folder. We have started to prepare a script for you, which you can find in bash_utilities/**organizse_from_tractoflow.sh**, which creates symlinks between your tractoflow results and a dwi_ml_ready folder. However, Tractoflow may have changed since we create this help, filenames could not correspond to your files. We encourage you to modify this script in your own project depending on your needs.
