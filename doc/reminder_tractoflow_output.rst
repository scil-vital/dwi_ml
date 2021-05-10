.. _ref_tractoflow_structure:


Tractoflow's output structure
=============================

Here is a description of tractoflow's typical output. For one subject:

.. code-block:: bash

    ./
    ├── Bet_DWI
    │   ├── {subjid}__b0_bet_mask.nii.gz
    │   ├── {subjid}__b0_bet.nii.gz
    │   └── {subjid}__dwi_bet.nii.gz
    ├── Bet_Prelim_DWI
    │   ├── {subjid}__b0_bet_mask_dilated.nii.gz
    │   ├── {subjid}__b0_bet_mask.nii.gz
    │   └── {subjid}__b0_bet.nii.gz
    ├── Bet_T1
    │   ├── {subjid}__t1_bet_mask.nii.gz
    │   └── {subjid}__t1_bet.nii.gz
    ├── Compute_FRF
    │   └── {subjid}__frf.txt
    ├── Crop_DWI
    │   ├── {subjid}__b0_cropped.nii.gz
    │   ├── {subjid}__b0_mask_cropped.nii.gz
    │   └── {subjid}__dwi_cropped.nii.gz
    ├── Crop_T1
    │   ├── {subjid}__t1_bet_cropped.nii.gz
    │   └── {subjid}__t1_bet_mask_cropped.nii.gz
    ├── Denoise_DWI
    │   └── {subjid}__dwi_denoised.nii.gz
    ├── Denoise_T1
    │   └── {subjid}__t1_denoised.nii.gz
    ├── DTI_Metrics
    │   ├── {subjid}__ad.nii.gz
    │   ├── {subjid}__evals_e1.nii.gz
    │   ├── {subjid}__evals_e2.nii.gz
    │   ├── {subjid}__evals_e3.nii.gz
    │   ├── {subjid}__evals.nii.gz
    │   ├── {subjid}__evecs.nii.gz
    │   ├── {subjid}__evecs_v1.nii.gz
    │   ├── {subjid}__evecs_v2.nii.gz
    │   ├── {subjid}__evecs_v3.nii.gz
    │   ├── {subjid}__fa.nii.gz
    │   ├── {subjid}__ga.nii.gz
    │   ├── {subjid}__md.nii.gz
    │   ├── {subjid}__mode.nii.gz
    │   ├── {subjid}__nonphysical.nii.gz
    │   ├── {subjid}__norm.nii.gz
    │   ├── {subjid}__pulsation_std_dwi.nii.gz
    │   ├── {subjid}__rd.nii.gz
    │   ├── {subjid}__residual_iqr_residuals.npy
    │   ├── {subjid}__residual_mean_residuals.npy
    │   ├── {subjid}__residual.nii.gz
    │   ├── {subjid}__residual_q1_residuals.npy
    │   ├── {subjid}__residual_q3_residuals.npy
    │   ├── {subjid}__residual_residuals_stats.png
    │   ├── {subjid}__residual_std_residuals.npy
    │   ├── {subjid}__rgb.nii.gz
    │   └── {subjid}__tensor.nii.gz
    ├── Eddy
    │   ├── {subjid}__bval_eddy
    │   ├── {subjid}__dwi_corrected.nii.gz
    │   └── {subjid}__dwi_eddy_corrected.bvec
    ├── Extract_B0
    │   └── {subjid}__b0.nii.gz
    ├── Extract_DTI_Shell
    │   ├── {subjid}__bval_dti
    │   ├── {subjid}__bvec_dti
    │   └── {subjid}__dwi_dti.nii.gz
    ├── Extract_FODF_Shell
    │   ├── {subjid}__bval_fodf
    │   ├── {subjid}__bvec_fodf
    │   └── {subjid}__dwi_fodf.nii.gz
    ├── FODF_Metrics
    │   ├── {subjid}__afd_max.nii.gz
    │   ├── {subjid}__afd_sum.nii.gz
    │   ├── {subjid}__afd_total.nii.gz
    │   ├── {subjid}__fodf.nii.gz
    │   ├── {subjid}__nufo.nii.gz
    │   ├── {subjid}__peak_indices.nii.gz
    │   └── {subjid}__peaks.nii.gz
    ├── N4_DWI
    │   └── {subjid}__dwi_n4.nii.gz
    ├── N4_T1
    │   └── {subjid}__t1_n4.nii.gz
    ├── Normalize_DWI
    │   ├── {subjid}__dwi_normalized.nii.gz
    │   └── {subjid}_fa_wm_mask.nii.gz
    ├── organisation.txt
    ├── PFT_Maps
    │   ├── {subjid}__interface.nii.gz
    │   ├── {subjid}__map_exclude.nii.gz
    │   └── {subjid}__map_include.nii.gz
    ├── Register_T1
    │   ├── {subjid}__output0GenericAffine.mat
    │   ├── {subjid}__output1InverseWarp.nii.gz
    │   ├── {subjid}__output1Warp.nii.gz
    │   ├── {subjid}__t1_mask_warped.nii.gz
    │   └── {subjid}__t1_warped.nii.gz
    ├── Resample_B0
    │   ├── {subjid}__b0_mask_resampled.nii.gz
    │   └── {subjid}__b0_resampled.nii.gz
    ├── Resample_DWI
    │   └── {subjid}__dwi_resampled.nii.gz
    ├── Resample_T1
    │   └── {subjid}__t1_resampled.nii.gz
    ├── Seeding_Mask
    │   └── {subjid}__seeding_mask.nii.gz
    ├── Segment_Tissues
    │   ├── {subjid}__map_csf.nii.gz
    │   ├── {subjid}__map_gm.nii.gz
    │   ├── {subjid}__map_wm.nii.gz
    │   ├── {subjid}__mask_csf.nii.gz
    │   ├── {subjid}__mask_gm.nii.gz
    │   └── {subjid}__mask_wm.nii.gz
    └── Tracking
        └── {subjid}__tracking.trk
