.. _ref_config_file:

3. Preparing the config file
============================

To create the hdf5 file, you will need a config file such as below. HDF groups will be created accordingly for each subject in the hdf5. Volume group will mimic nifti files (you may concatenate many nifti files in a single group) and streamline groups will mimic tractogram files (you may concatenate many .trk or .tck files in a single group, for instance you could concatenate many bundles per subject).

This HDF5 will be used during training. For instance, if your models needs an input volume and the streamlines as target, you might need one group for each.

The HDF5 can also be used during tracking if you intend to track from a generative model. You might want to include the tracking masks here to have them ready for when your model will be trained, as another volume group.

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

General group attributes:

    - The group **names** could be 'input_volume', 'target_volume', 'target_directions', or anything.

        - Make sure your training scripts and your model's batch_sampler use the same keys.

    - The groups **"files"** must exist in every subject folder inside dwi_ml_ready (except if option 'enforce_files_presence is set to False).

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
