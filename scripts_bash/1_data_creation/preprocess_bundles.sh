echo "Processing bundles..."
  bundlespath="$original_dataset/bundles/$subjid"
  if [ ! -d "$bundlespath" ]; then
    echo "ERROR! Bundles for subject $subjid not found!"
  else

    tmp_folder="$dwi_ml_ready_folder/$subjid/bundles/tmp"
    if [ -d "$tmp_folder" ]; then
      rm -rf "$tmp_folder"
    fi
    mkdir "$tmp_folder"

    ### Check if we need to run TractSeg
    run_tractseg=false
    for input_trk in "$bundlespath"/tracts/*.trk; do
      bundlename=$(basename "$input_trk")
      bundlename=$(echo "$bundlename" | cut -f 1 -d '.')
      output_endpoints_head_mask="$dwi_ml_ready_folder/$subjid/masks/bundles/${subjid}_${bundlename}_head.nii.gz"
      output_endpoints_tail_mask="$dwi_ml_ready_folder/$subjid/masks/bundles/${subjid}_${bundlename}_tail.nii.gz"
      if [ ! -f "$output_endpoints_head_mask" ] || [ ! -f "$output_endpoints_tail_mask" ]; then
        run_tractseg=true
        break
      fi
    done

    ### Run TractSeg to get enpoints masks ###
    if [ "$run_tractseg" = true ]; then
      echo "Running TractSeg..."
      TractSeg -i "$subj_folder/data.nii.gz" -o "$tmp_folder" --bvals "$subj_folder/bvals" --bvecs "$subj_folder/bvecs" --raw_diffusion_input --brain_mask "$subj_folder/nodif_brain_mask.nii.gz" --output_type endings_segmentation --csd_type csd_msmt_5tt
    fi
    tractseg_endpoints="$tmp_folder/endings_segmentations"

    ### Convert TRK to TCK ###
    for input_trk in "$bundlespath"/tracts/*.trk; do
      bundlename=$(basename "$input_trk")
      bundlename=$(echo "$bundlename" | cut -f 1 -d '.')
      output_tck="$dwi_ml_ready_folder/$subjid/bundles/${subjid}_${bundlename}.tck"
      if [ ! -f "$output_tck" ]; then
        echo "Processing $input_trk..."
        scil_convert_tractogram.py "$input_trk" "$output_tck" --reference "$dwi_final"
      else
        echo "Bundle $(basename "$output_tck") already exists, skipping..."
      fi

      output_bundle_mask="$dwi_ml_ready_folder/$subjid/masks/bundles/${subjid}_${bundlename}.nii.gz"
      if [ ! -f "$output_bundle_mask" ]; then
        # TODO: Fix this once the scripts have been moved to scilpy-public
        deactivate
        source ~/.virtualenvs/scilpy/bin/activate

        # Compute bundle mask
        ref_fa=$dwi_ml_ready_folder/$subjid/dwi/${subjid}_fa.nii.gz
        scil_compute_density_map_from_streamlines.py "$output_tck" "$ref_fa" "$output_bundle_mask" --binary

        deactivate
        source ~/.virtualenvs/learn2track/bin/activate
      fi

      output_endpoints_head_mask="$dwi_ml_ready_folder/$subjid/masks/bundles/${subjid}_${bundlename}_head.nii.gz"
      output_endpoints_tail_mask="$dwi_ml_ready_folder/$subjid/masks/bundles/${subjid}_${bundlename}_tail.nii.gz"
      if [ ! -f "$output_endpoints_head_mask" ] || [ ! -f "$output_endpoints_tail_mask" ]; then
        # Flip TractSeg endpoints maps (computed on the original LAS diffusion)
        head_mask=$tractseg_endpoints/${bundlename}_b.nii.gz
        tail_mask=$tractseg_endpoints/${bundlename}_e.nii.gz
        mrconvert "$head_mask" "$output_endpoints_head_mask" -strides 1,2,3
        mrconvert "$tail_mask" "$output_endpoints_tail_mask" -strides 1,2,3
      fi
    done

    # Merge all bundle masks to use as a WM mask
    wm_mask="$dwi_ml_ready_folder/$subjid/masks/${subjid}_wm.nii.gz"
    if [ ! -f "$wm_mask" ]; then
      echo "Merging all bundle masks to build a WM mask..."
      all_bundles=$(find "$dwi_ml_ready_folder"/"$subjid"/masks/bundles/* -type f -not -name "*_head.nii.gz" -not -name "*_tail.nii.gz")
      scil_mask_math.py union $all_bundles "$wm_mask"
    fi

    rm -rf "$tmp_folder"
  fi