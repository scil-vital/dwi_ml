
FOLDER_DESCRIPTION = (
    """=== Expected folder structure:
{dataset_name}
| data                  ==========> Your original data folder
    | {subject_id}
        | data.nii.gz
        | bvals
        | bvecs
        | nodif_brain_mask.nii.gz
| raw                  ==========> Created by 1_convert_original_to_raw.sh
    | {subject_id}
        | dwi
            | {subject_id}_dwi.nii.gz
            | {subject_id}_dwi.bvals
            | {subject_id}_dwi.bvecs
            | {subject_id}_fa.nii.gz              # NOT USED HERE
        | bundles
            | {subject_id}_bundle_1.tck
            | ...
            | {subject_id}_bundle_{N}.tck
            *OR*
            | {subject_id}_wholebrain.tck
        | masks
            | {subject_id}_normalization.nii.gz
    | ...

""")

BIDS_DESCRIPTION = (
    """=== Expected BIDS structure:                                                             # POUR COMMENCER À SE PRÉPARER AUX BIDS
{dataset_name}
| raw
    | {subject_id1}_dwi.nii.gz
    | {subject_id2}_dwi.nii.gz
    ...
| derivatives_bundles1
    | {subject_id1}_bundle_1.tck
    | {subject_id2}_bundle_1.tck
    ...
| derivatives_mask
    | {subject_id1}_wm.nii.gz
    ...
    """
)

CONFIG_DECRIPTION = (
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
- bval: If provided, keep only the given b-value (and b0s). (+/- 50)                                                # Copié le (+/- 50) de ANTOINE. Vérifier si vrai dans tous les codes
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