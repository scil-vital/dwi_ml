.. _ref_config_file:

3. Preparing the config file
============================

To create the hdf5 file, you will need a config file such as below.

.. code-block:: bash

    {
        "input_group": {
            "type": "volume",
            "files": ["dwi/dwi_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz"]
            "standardization": "all"
             },
        "good_streamlines": {
            "type": "streamlines",
            "files": ["tractograms/bundle1.trk", "tractograms/wholebrain.trk"]
             }
        "bad_streamlines": {
            "type": "streamlines",
            "files": ["bad_tractograms/ALL"]
             }
    }

- The group **names** could be 'input_volume', 'target_volume', 'target_directions', or anything. Make sure your training scripts and your model's batch_sampler use the same keys.

- The groups **"files"** must exist in every subject folder inside dwi_ml_ready.

    - There is the possibility to add a wildcard (*) that will be replaced by the subject's id while loading. Ex: anat/\*__t1.nii.gz would become anat/subjX__t1.nii.gz.
    - For streamlines, there is the possibility to use 'ALL' to load all tractograms present in a folder. Ex: "files": ["good_bundles/ALL", "bad_bundles/ALL"]
    - The files from each group will be concatenated in the hdf5 (either as a final volume or as a final tractogram).

- The groups **"type"** must be recognized in dwi_ml. Currently, accepted datatype are:

    - 'volume': for instance, a dwi, an anat, mask, t1, fa, etc.
    - 'streamlines': for instance, a .trk, .tck file (anything accepted by Dipy's Stateful Tractogram).

- The groups **"standardization"** must be of of

    - "all", to apply standardization (normalization) to the final (concatenated) file
    - "independent", to apply it independently on the last dimension of the data (ex, for a fODF, it would apply it independently on each SH).
    - "per_file", to apply it independently on each file included in the group.
    - "none", to skip this step (ex: for binary masks, which must stay binary).

    ****A note about data standardization**

    Data is standardized (normalized) during data creation: data = (data - mean) / std.

    If all voxel were to be used, most of them would probably contain the background of the data, bringing the mean and std probably very close to 0. Thus, non-zero voxels only are used to compute the mean and std, or voxels inside the provided mask if any.

    In the latest case, voxels outside the mask could have been set to NaN, but a test with the b0 as a mask showed that some streamlines had points outside the mask (probably due to data interpolation or to the skull-stripping technique of the b0 mask). The safer choice, chosen in dwi_ml, was to simply modify all voxels [ data = (data - mean) / std ], even voxels outside the mask.

    Mask is provided using --std_mask in the script create_hdf5_dataset.py.
