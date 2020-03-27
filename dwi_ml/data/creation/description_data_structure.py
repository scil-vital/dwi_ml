
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
| processed_{experiment_name}
    (depends if hdf5 or BIDS. see PROCESSED_DESCRIPTION_HDF5)
"""

PROCESSED_DESCRIPTION_HDF5 = """
| processed_{experiment_name}  ===========> Created by your run_project.sh
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
    """=== Expected json config file structure:
{
    "bval": 1000,
    "minimum_length_mm": 10.0,
    "step_size_mm": 1.0,
    "subject_ids": [
        "subject1",...
    ]
    "bundles": {
        "bundle_1": {
            "clustering_threshold_mm": 6.0,
            "removal_distance_mm": 2.0
        },
        ...
    }
}

** Config file parameters:
- sh_order: Order of the spherical harmonics to fit onto the signal.
  If zero, use only the b0-attenuated dwi signal.
- bval: If provided, keep only the given b-value (and b0s).
- minimum_length_mm: Discard streamlines shorter than this length (in mm).
- step_size_mm: Resample streamlines to have the given step size between
  every point (in mm). *Important: Note that if no step size is defined, the
  streamlines will be compressed.
- subject_ids: List of subjects to include in the dataset.
- bundles: List of bundles to include in the dataset. If none is given, we will
  still look into the `bundles` folder, but we will be looking for wholebrain
  tractograms.

** Bundle-specific subsampling parameters
- name: The name of the bundle that should be included in the dataset.
- clustering_threshold_mm: Threshold used to cluster streamlines before
  computing distance matrix (in mm).
- removal_distance_mm: Streamlines with an average distance smaller than this
  will be reduced to a single streamline (in mm).

""")
