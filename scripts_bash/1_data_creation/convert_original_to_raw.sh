################################################################################
# This script is used to precess the original HCP data and organize it into the
# expected "raw" folder architecture, flipping the DWI and bvecs from LAS to RAS
# and creating all necessary masks.
# the raw/ folder will be created alongside the original/ folder.
#
# Args:
#   working_folder : Path to the working folder that contains the original/
#                    folder with the HCP105 dataset.
#                    original/ should contain the folders data/
#                    (from Glasser, 2013) and bundles/ (from TractSeg), each
#                    containing all subject folders.
#   subjects : A .txt file listing all the subjects ids to process, as expected
#              in the data/ and bundles/ folders.
################################################################################
if [ ! -d "$1" ]; then
  echo "Invalid working_folder argument! : $1"
  exit
fi
if [ ! -f "$2" ]; then
  echo "Invalid subjects .txt file! : $2"
  exit
fi
original_dataset=$1/original
raw_folder=$1/raw
subjects=$2

if [ ! -d "$raw_folder" ]; then
  mkdir "$raw_folder"
fi

for subid in $(<"$subjects"); do
  echo "Processing subject $subid"

  if [ ! -d "$raw_folder/$subid/" ]; then
    mkdir "$raw_folder/$subid"
  fi

  if [ ! -d "$raw_folder/$subid/bundles" ]; then
    mkdir "$raw_folder/$subid/bundles"
  fi

  if [ ! -d "$raw_folder/$subid/dwi" ]; then
    mkdir "$raw_folder/$subid/dwi"
  fi

  if [ ! -d "$raw_folder/$subid/masks" ]; then
    mkdir "$raw_folder/$subid/masks"
  fi

  if [ ! -d "raw/$subid/masks/bundles" ]; then
    mkdir "raw/$subid/masks/bundles"
  fi

  echo "Processing diffusion..."
  datapath="$original_dataset/data/$subid"

  if [ ! -d "$datapath" ]; then
    echo "ERROR! Diffusion for subject $subid not found! : $datapath"
  else
    orig_diff="$datapath/data.nii.gz"
    orig_bvals="$datapath/bvals"
    orig_bvecs="$datapath/bvecs"

    tmp_folder="$raw_folder/$subid/dwi/tmp"
    if [ -d "$tmp_folder" ]; then
      rm -rf "$tmp_folder"
    fi
    mkdir "$tmp_folder"

    ### Process dwi ###
    dwi_final=$raw_folder/$subid/dwi/${subid}_dwi.nii.gz
    bvals="$raw_folder/$subid/dwi/${subid}_dwi.bvals"
    bvecs="$raw_folder/$subid/dwi/${subid}_dwi.bvecs"

    if [ ! -f "$dwi_final" ]; then
      # DWI+bvecs needs to be flipped, DWI+bvals+bvecs need to be filtered for b=0,1000

      echo "Flipping bvecs (LAS -> RAS)..."
      scil_flip_gradients.py "$orig_bvecs" "$tmp_folder/${subid}_dwi.bvecs" x --fsl
      orig_bvecs=$tmp_folder/${subid}_dwi.bvecs

      echo "Flipping diffusion (LAS -> RAS)..."
      output_diff=$tmp_folder/dwi_orig_flipped.nii.gz
      mrconvert "$orig_diff" "$output_diff" -strides 1,2,3,4
      tmp_diff=$output_diff

      echo "Extracting b=1000 shell..."
      output_diff=$tmp_folder/dwi.nii.gz
      scil_extract_dwi_shell.py "$tmp_diff" "$orig_bvals" "$orig_bvecs" 0 1000 "$output_diff" "$bvals" "$bvecs" -t 20
      tmp_diff=$output_diff
    else
      # Re-use processed DWI
      tmp_diff=$dwi_final
    fi

    ### Compute FA ###
    fa=$raw_folder/$subid/dwi/${subid}_fa.nii.gz
    if [ ! -f "$fa" ]; then
      ref_mask=$datapath/nodif_brain_mask.nii.gz
      scil_compute_dti_metrics.py "$tmp_diff" "$bvals" "$bvecs" --mask "$ref_mask" --not_all --fa "$fa"
    fi

    if [ ! -f "$dwi_final" ]; then
      echo "Computing approximate WM mask for DWI normalization..."
      wm_mask=$tmp_folder/${subid}_wm.nii.gz

      # Get WM mask from FA threshold
      mrthreshold "$fa" "$wm_mask" -abs 0.4 -nthreads 1

      echo "Normalizing DWI..."
      dwinormalise "$tmp_diff" "$wm_mask" "$dwi_final" -fslgrad "$bvecs" "$bvals" -nthreads 1
    else
      echo "File $(basename "$dwi_final") already exists"
    fi

    rm -rf "$tmp_folder"
  fi

  echo "Processing bundles..."
  bundlespath="$original_dataset/bundles/$subid"
  if [ ! -d "$bundlespath" ]; then
    echo "ERROR! Bundles for subject $subid not found!"
  else

    tmp_folder="$raw_folder/$subid/bundles/tmp"
    if [ -d "$tmp_folder" ]; then
      rm -rf "$tmp_folder"
    fi
    mkdir "$tmp_folder"

    ### Check if we need to run TractSeg
    run_tractseg=false
    for input_trk in "$bundlespath"/tracts/*.trk; do
      bundlename=$(basename "$input_trk")
      bundlename=$(echo "$bundlename" | cut -f 1 -d '.')
      output_endpoints_head_mask="$raw_folder/$subid/masks/bundles/${subid}_${bundlename}_head.nii.gz"
      output_endpoints_tail_mask="$raw_folder/$subid/masks/bundles/${subid}_${bundlename}_tail.nii.gz"
      if [ ! -f "$output_endpoints_head_mask" ] || [ ! -f "$output_endpoints_tail_mask" ]; then
        run_tractseg=true
        break
      fi
    done

    ### Run TractSeg to get enpoints masks ###
    if [ "$run_tractseg" = true ]; then
      echo "Running TractSeg..."
      TractSeg -i "$datapath/data.nii.gz" -o "$tmp_folder" --bvals "$datapath/bvals" --bvecs "$datapath/bvecs" --raw_diffusion_input --brain_mask "$datapath/nodif_brain_mask.nii.gz" --output_type endings_segmentation --csd_type csd_msmt_5tt
    fi
    tractseg_endpoints="$tmp_folder/endings_segmentations"

    ### Convert TRK to TCK ###
    for input_trk in "$bundlespath"/tracts/*.trk; do
      bundlename=$(basename "$input_trk")
      bundlename=$(echo "$bundlename" | cut -f 1 -d '.')
      output_tck="$raw_folder/$subid/bundles/${subid}_${bundlename}.tck"
      if [ ! -f "$output_tck" ]; then
        echo "Processing $input_trk..."
        scil_convert_tractogram.py "$input_trk" "$output_tck" --reference "$dwi_final"
      else
        echo "Bundle $(basename "$output_tck") already exists, skipping..."
      fi

      output_bundle_mask="$raw_folder/$subid/masks/bundles/${subid}_${bundlename}.nii.gz"
      if [ ! -f "$output_bundle_mask" ]; then
        # TODO: Fix this once the scripts have been moved to scilpy-public
        deactivate
        source ~/.virtualenvs/scilpy/bin/activate

        # Compute bundle mask
        ref_fa=$raw_folder/$subid/dwi/${subid}_fa.nii.gz
        scil_compute_density_map_from_streamlines.py "$output_tck" "$ref_fa" "$output_bundle_mask" --binary

        deactivate
        source ~/.virtualenvs/learn2track/bin/activate
      fi

      output_endpoints_head_mask="$raw_folder/$subid/masks/bundles/${subid}_${bundlename}_head.nii.gz"
      output_endpoints_tail_mask="$raw_folder/$subid/masks/bundles/${subid}_${bundlename}_tail.nii.gz"
      if [ ! -f "$output_endpoints_head_mask" ] || [ ! -f "$output_endpoints_tail_mask" ]; then
        # Flip TractSeg endpoints maps (computed on the original LAS diffusion)
        head_mask=$tractseg_endpoints/${bundlename}_b.nii.gz
        tail_mask=$tractseg_endpoints/${bundlename}_e.nii.gz
        mrconvert "$head_mask" "$output_endpoints_head_mask" -strides 1,2,3
        mrconvert "$tail_mask" "$output_endpoints_tail_mask" -strides 1,2,3
      fi
    done

    # Merge all bundle masks to use as a WM mask
    wm_mask="$raw_folder/$subid/masks/${subid}_wm.nii.gz"
    if [ ! -f "$wm_mask" ]; then
      echo "Merging all bundle masks to build a WM mask..."
      all_bundles=$(find "$raw_folder"/"$subid"/masks/bundles/* -type f -not -name "*_head.nii.gz" -not -name "*_tail.nii.gz")
      scil_mask_math.py union $all_bundles "$wm_mask"
    fi

    rm -rf "$tmp_folder"
  fi
done
