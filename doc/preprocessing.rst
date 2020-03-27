.. _ref_preprocessing:

Preprocessing your data
=======================

Preprocessing diffusion data
****************************

See the `SCIL's documentation <https://scil-documentation.readthedocs.io/en/latest/?badge=latest>`_ for more information on how to preprocess your data using tractoflow. Then, see our :ref:`ref_organization` tab to learn to organize everything as we expect.

Preprocessing bundles
*********************

Here is an example of steps that could be useful for you to preprocess bundles. We consider that tractoflow has already been ran.

Conversions
    Here is how to convert from trk to tck:

    .. code-block:: bash

        scil_convert_tractogram.py TRK_FILE TCK_OUT_NAME \
            --reference processed/SUBJ/some_ref.nii.gz

Separating your tractogram into bundles
    One possible technique to create bundle is to simply regroup the streamlines that are close. See ``scil_compute_qbx.py``.

    For a clustering based on atlases, Dipy offers Recobundles, or you can use scilpy's version RecobundlesX, which is a little different. You will need bundle models and their associated json file (here named hcp_models). In this example, we use Tractoflow's tracking result.

    .. code-block:: bash

            scil_recognize_multi_bundles.py Tracking/SUBJ__tracking.trk \
                hcp_models/config_file.json hcp_models/* tmp_folder

    You could also check our bundle specific tools such as

Bundle masks
    Here is how to create a mask of voxels touched by a bundle:

    .. code-block:: bash

        scil_compute_density_map_from_streamlines.py BUNDLE.tck \
            preprocessed/subj/DTI_Metrics/SUBJ__fa.nii.gz OUT_NAME --binary

Inverting endpoints masks
    You can use mrtrix to invert masks.

    .. code-block:: bash

        # ?? TO CLEAN
        mrconvert "$head_mask" "$output_endpoints_head_mask" -strides 1,2,3

Merge bundles
    Here is how you can merge bundles together:

    .. code-block:: bash

        scil_mask_math.py union ALL_BUNDLES preprocessed/SUBJ/some_mask.nii.gz

Tractseg
    `Tractseg <https://github.com/MIC-DKFZ/TractSeg>`_ is one of the most used published techniques using machine learning for diffusion. If you want to compare your work with theirs, you might want to use their bundles. Here is how to use it:

    .. code-block:: bash

        TractSeg -i YOUR_DATA -o OUT_NAME --bvals original/SUBJ/bval \
            --bvecs original/SUBJ/bvec --raw_diffusion_input \
            --brain_mask preprocessed/SUBJ/some_mask \
            --output_type endings_segmentation --csd_type csd_msmt_5tt