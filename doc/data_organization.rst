.. _ref_organization:

2. Organizing your data
=======================

This is how your data should be organized before trying to load your data as a hdf5 file. This structure should hold wether you work with hdf5 or BIDS.

**tractoflow_output:**

    If you used tractoflow to preprocess your data, you may organize automatically the dwi_ml_ready folder (see below). We have started to prepare a script for you, which you can find in please_copy_and_adapt/**organizse_from_tractoflow.sh**, which creates symlinks between your tractoflow results and a dwi_ml_ready folder. We encourage you to modify this script in your own project depending on your needs.

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
