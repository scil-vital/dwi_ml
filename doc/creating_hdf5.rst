Creating the hdf5 file
======================

You will use the create_hdf5_dataset.py script to create a hdf5 file. You need to prepare config files to use this script (see lower). This is the first step found in please_copy_and_adapt/run_project.sh.

Config files
************

**Group config file**

Expected json config for the groups in your hdf5:

.. code-block:: bash

    {
        "group1": ["file1.nii.gz", "file2.nii.gz", ...],
        "group2": ["file1.nii.gz"]
    }

The group names could be 'input_volume', 'target_volume', for example. Make sure your training script calls the same keys.

The filenames could be 'anat/dwi_tractoflow.nii.gz' for example. Must exist in every subject folder inside dwi_ml_ready.
