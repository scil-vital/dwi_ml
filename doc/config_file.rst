3. Preparing the config file
============================

To create the hdf5 file, you will need a config file such as below.

.. code-block:: bash

    {
        "group1": {
            "type": "volume",
            "files": ["dwi/dwi_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz"]
             },
        "group2": {
            "type": "streamlines",
            "files": ["bundles/bundle1.trk", "bundles/bundle2.trk"]
             }
    }

- The group names could be 'input_volume', 'target_volume', 'target_directions', or anything. Make sure your training scripts and your model's batch_sampler use the same keys.
- The groups 'files' must exist in every subject folder inside dwi_ml_ready.

    - There is the possibility to add a wildcard (*) that will be replaced by the subject's id while loading. Ex: anat/\*__t1.nii.gz would become anat/subjX__t1.nii.gz.
    - For streamlines, there is the possibility to use 'ALL' to load all bundles present. Ex: "files": ["good_bundles/ALL", "bad_bundles/ALL"]
    - The files from each group will be concatenated in the hdf5 (either as a final volume or as a final tractogram).
- The groups 'type' must be recognized in dwi_ml. Currently, accepted datatype are:

    - 'volume': for instance, a dwi, an anat, mask, t1, fa, etc.
    - 'streamlines': for instance, a .trk, .tck file (anything accepted by Dipy's Stateful Tractogram).
