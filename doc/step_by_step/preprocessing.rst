.. _ref_preprocessing:

1. Preprocessing your data (using scilpy)
=========================================

Running the whole tractography process
**************************************

We suggest using scilpy's `tractoflow <https://tractoflow-documentation.readthedocs.io/en/latest/>`_ to preprocess dwi data and create tractograms.

Obtaining clean bundles
***********************

Here is an example of steps that could be useful to preprocess bundles. Here, we consider that tractoflow has already been ran.

Separating your tractogram into bundles
'''''''''''''''''''''''''''''''''''''''

You might want to separate some good bundles from the whole-brain tractogram (ex, for bundle-specific tractography algorithms, BST, or simply to ensure that you train your ML algorithm on true-positive streamlines only).

One possible technique to create bundles is to simply regroup the streamlines that are close based on some metric (ex, the MDF). See `scil_compute_qbx.py <https://github.com/scilus/scilpy/blob/master/scripts/scil_compute_qbx.py>`_.

However, separating data into known bundles (from an atlas) is probably a better way to clean your tractogram and to remove all possibly false positive streamlines. For a clustering based on atlases, Dipy offers Recobundles, or you can use scilpy's version RecobundlesX, which is a little different. You will need bundle models and their associated json file. You may check `scilpy's doc <https://scil-documentation.readthedocs.io/en/latest/our_tools/recobundles.html>`_ RecobundlesX tab for a basic bash script example.


Tractseg
''''''''

`Tractseg <https://github.com/MIC-DKFZ/TractSeg>`_ is one of the most used published techniques using machine learning for diffusion. If you want to compare your work with theirs, you might want to use their bundles. Here is how to use it:

    .. code-block:: bash

        TractSeg -i YOUR_DATA -o OUT_NAME --bvals original/SUBJ/bval \
            --bvecs original/SUBJ/bvec --raw_diffusion_input \
            --brain_mask preprocessed/SUBJ/some_mask \
            --output_type endings_segmentation --csd_type csd_msmt_5tt

Other tools and tricks
***********************


Tractogram conversions
''''''''''''''''''''''

Here is how to convert from trk to tck:

    .. code-block:: bash

        scil_convert_tractogram.py TRK_FILE TCK_OUT_NAME \
            --reference processed/SUBJ/some_ref.nii.gz

Bundle masks
''''''''''''

Here is how to create a mask of voxels touched by a bundle:

    .. code-block:: bash

        scil_compute_density_map_from_streamlines.py BUNDLE.tck \
            preprocessed/subj/DTI_Metrics/SUBJ__fa.nii.gz OUT_NAME --binary

Merging bundles
'''''''''''''''

Here is how you can merge bundles together:

    .. code-block:: bash

        scil_mask_math.py union ALL_BUNDLES preprocessed/SUBJ/some_mask.nii.gz
