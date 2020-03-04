
FOLDER_DESCRIPTION = """
=== Expected folder structure. This structure should hold wether you
    work with hdf5 or BIDS.

{database_name}
| original      ==========> Your original data folder. Necessary for tractoflow
    | {subject_id}
        | dwi.nii.gz
        | bval
        | bvec
        | t1.nii.gz
| preprocessed  ======> No matter how you preprocess your data, keep results
                   here. Ex: tractoflow + any other technique to get your
                   bundles.
    | Ex: Tractoflow
    | Ex: Bundles from Recobundles
| dwi_ml_ready  =====> If you used tractoflow, you can use organize_dwi_ml_ready.sh
                       to arrange your data.
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
"""

PROCESSED_DESCRIPTION_HDF5 = """
| processed_{experiment_name}  ===========> Created by your create_dataset.sh
                                      (only saved if option --save_intermediate)
    | {subject_id}
        | input
            | {subject_id}_{input1}.nii.gz  # Ex: fODF (unnormalized)
            ...
            | {subject_id}_{inputN}.nii.gz
            | {subject_id}_model_input.nii.gz   # Final input = all inputs,
                                                #  normalized, concateanted.
        | target
            | {subject_id}_{target1}.tck  # Ex: bundle1
"""

CONFIG_DESCRIPTION = (
    """=== Expected json config for the groups in your hdf5:
{
    "group1": ["file1.nii.gz", "file2.nii.gz", ...],
    "group2": ["file1.nii.gz"]
}

For example, the group names could be 'input_volume', 'target_volume', etc. 
Make sure your training script calls the same keys.""")
