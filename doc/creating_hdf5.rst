Creating the hdf5 file
======================

You will use the create_dataset.sh script to create a hdf5 file. You need to prepare config files to use this script.

Config files
************

**Group config file**

Expected json config for the groups in your hdf5:

.. code-block:: bash

    {
        "group1": ["file1.nii.gz", "file2.nii.gz", ...],
        "group2": ["file1.nii.gz"]
    }

For example, the group names could be 'input_volume', 'target_volume', etc.
Make sure your training script calls the same keys.

**Bundles config file**

Expected json config for the bundles in your hdf5:

.. code-block:: bash

    {
        "bundle1": [clustering_threshold_mm, removal_distance_mm],
        "bundle2": []
    }

For example, the group names could be 'input_volume', 'target_volume', etc.
Make sure your training script calls the same keys.