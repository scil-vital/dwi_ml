.. _ref_organization:

Organizing your data
====================

Folder division
***************

This is how your data should be organized before trying to load your data as a hdf5 file. This structure should hold wether you work with hdf5 or BIDS.

**Original:**

    Your original data folder. This is facultative but you probably have already organized your data like that if you ran tractoflow. For more information on data organization for tractoflow, please check `Tractoflow's Input Structure <https://tractoflow-documentation.readthedocs.io/en/latest/pipeline/input.html>`_.

**tractoflow_output:**

    If you used tractoflow to preprocess your data, you may organize automatically the dwi_ml_ready folder (see below). We have started to prepare a script for you, which you can find in please_copy_and_adapt/**01_organizse_from_tractoflow.sh**, which creates symlinks between your tractoflow results and a dwi_ml_ready folder. We encourage you to modify this script in your own project depending on your needs. You can find `here <./reminder_tractoflow_output.rst>`_ a description of tractoflow's typical output.

**dwi_ml_ready**

    This folder is the most important one and must be organized in this exact way to be able to load the data as a hdf5 using our script create_hdf5_dataset.py. An example of use can be found in scripts_python/**create_hdf5_dataset.sh**.

**processed**

    A processed folder will be created automatically. See next section for description.

Example
*******

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
