.. _ref_organization:

Organizing your data
====================

Folder division
***************

This is how your data should be organized before trying to load your data as a hdf5 file. This structure should hold wether you work with hdf5 or BIDS.

**Original:**

    Your original data folder. This is facultative but you probably have already organized your data like that if you ran tractoflow. For more information on data organization for tractoflow, please check `Tractoflow's Input Structure <https://tractoflow-documentation.readthedocs.io/en/latest/pipeline/input.html>`_.

**Preprocessed:**

    No matter how you preprocess your data, please keep the results in a "preprocessed" folder. Ex: tractoflow + any other technique to get your bundles.

    ** Note. If you used tractoflow and have kept the results in preprocessed, you can organize automatically the dwi_ml_ready folder. We have started to prepare a script for you, which you can find in please_copy_and_adapt/**01_organizse_from_tractoflow.sh**, which creates symlinks between your tractoflow results and a dwi_ml_ready folder. We encourage you to modify this script in your own project depending on your needs. You can find `here <./reminder_tractoflow_output.rst>`_ a description of tractoflow's typical output.

**dwi_ml_ready**

    This folder is the most important one and must be organized in this exact way to be able to load the data as a hdf5 using our script create_hdf5_dataset.py. An example of use can be found in please_copy_and_adapt/**02_create_dataset.sh**.

**processed**

    A processed folder will be created automatically. See next section for description.

Example
*******

.. code-block:: bash

    {database_name}
    | original
        | {subject_id}
            | dwi.nii.gz
            | bval
            | bvec
            | t1.nii.gz
    | preprocessed
        | {subject_id}
            | Ex: Tractoflow folders
            | Ex: a folder Bundles with bundles from Recobundles
    | dwi_ml_ready  =====>
        | {subject_id}
            | anat
                | {subject_id}_t1.nii.gz
                | {subject_id}_wm_map.nii.gz
            | dwi
                | {subject_id}_dwi_preprocessed.nii.gz
                | {subject_id}_bval_preprocessed
                | {subject_id}_bvec_preprocessed
                | {subject_id}_fa.nii.gz
            | bundles
                | {subject_id}_{bundle1}.tck
            | masks
                | {subject_id}_wm.nii.gz
                | bundles
                    | {subject_id}_{bundle1}.nii.gz
                | endpoints
                    | {subject_id}_{bundle1}.nii.gz
                    OR
                    | {subject_id}_{bundle1}_heads.nii.gz
                    | {subject_id}_{bundle1}_tails.nii.gz
        | ...


Once the hdf5 will have been created (using create_hdf5_dataset.py), if you chose the --save_intermediate option, a processed folder with the following structure will be created:

.. code-block:: bash

    | {database_name}
        | processed_{experiment_name}
            | {subject_id}
                | input
                    | {subject_id}_{input1}.nii.gz  # Ex: fODF (unnormalized)
                    ...
                    | {subject_id}_{inputN}.nii.gz
                    | {subject_id}_model_input.nii.gz   # Final input = all inputs,
                                                        #  normalized, concateanted.
                | target
                    | {subject_id}_{target1}.tck  # Ex: bundle1
