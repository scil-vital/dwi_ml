#!/bin/bash

###############################################################################
# Your tree should look like:                                                 #
# derivatives                                                                 #
#    ├── original (ex, tractoflow input)                                      #
#    └── preprocessed (ex, tractoflow output + recobundle output)             #
#    └── dwi_ml_ready: will be created now.                                   #
#                                                                             #
# This script will create symlinks in dwi_ml_ready, pointing to your data     #
# from tractoflow for each subject:                                           #
#       └── dwi:                                                              #
#           └── dwi_tractoflow: Resample/dwi_resample                         #
#               (It is the last dwi output from tractoflow.)                  #
#           └── bval_tractoflow: Eddy/bval_eddy                               #
#           └── bvec_tractoflow: Eddy/bvec_eddy                               #
#           └── fa: DTI_metrics/fa                                            #
#       └── anat :                                                            #
#           └── t1_tractoflow: Register_T1/t1_warp                            #
#               (It is the last T1 output from tractoflow.)                   #
#           └── wm_map: Segment_Tissues/map_wm                                #
#       └── masks :                                                           #
#           └── wm: egment_Tissues/mask_wm                                    #
#       └── bundles  :                                                        #
#           └── tractoflow_wholebrain: Tracking/tracking                      #
#                                                                             #
# If you need something else for your model, you can modify this script.      #
#                                                                             #
# See our doc for more information                                            #
# (https://dwi-ml.readthedocs.io/en/latest/data_organization.html#ref-organization).
# We suppose that you have a "preprocessed" folder that contains RecobundlesX #
# results folder for each subject.                                            #
###############################################################################

# =====================================#
#  VARIABLES TO BE DEFINED BY THE USER #
# =====================================#
# - database_folder = Path to the working folder that contains the original/
#     folder. Will eventually contain dwi_ml_ready/.
# - subjects = The list of ALL subjects. You may choose later which ones will be
#     in your training set/validation set/testing set. One subject per line.
database_folder=$1   # YOUR WORKING FOLDER
subject_list=$2      # ex, SUBJECTS.txt

# =====================================#
#            MAIN SCRIPT               #
# =====================================#
# Cleaning path name
database_folder=$(realpath $database_folder)

# Checking if inputs exist
if [ ! -d $database_folder ]; then
  echo "Database not found! ($database_folder)!"
  exit
fi
if [ ! -d $database_folder/preprocessed ]; then
  echo "There is no preprocessed folder in your database! ($database_folder)"
  exit
fi
if [ ! -f $subject_list ]; then
  echo "Invalid subjects txt file! ($subject_list)"
  exit
fi

# Preparing dataset subfolders
dwi_ml_ready_folder=$database_folder/dwi_ml_ready
if [ ! -d "$dwi_ml_ready_folder" ]; then
  mkdir "$dwi_ml_ready_folder"
else
  echo "The dwi_ml_ready folder already exists!!! Please delete it first."
  exit 1
fi

# Reorganizing all subjects
echo "Checks passed. Now reorganizing subjects"
subjects=$(<$subject_list)
for subjid in $subjects
do
  echo "Reorganizing subject $subjid"
  subj_preprocessed_folder=$database_folder/preprocessed/$subjid
  subj_folder=$dwi_ml_ready_folder/$subjid
  mkdir $subj_folder
  mkdir $subj_folder/anat
  mkdir $subj_folder/dwi
  mkdir $subj_folder/masks
  mkdir $subj_folder/bundles

  # dwi:
  if [ ! -f $subj_preprocessed_folder/Resample_DWI/${subjid}__dwi_resampled.nii.gz ]; then echo "Subject's DWI not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Resample_DWI/${subjid}__dwi_resampled.nii.gz $subj_folder/dwi/dwi_tractoflow.nii.gz
  if [ ! -f $subj_preprocessed_folder/Eddy/${subjid}__bval_eddy ]; then echo "Subject's bval not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Eddy/${subjid}__bval_eddy $subj_folder/dwi/bval_tractoflow
  if [ ! -f $subj_preprocessed_folder/Eddy/${subjid}__dwi_eddy_corrected.bvec ]; then echo "Subject's bvec not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Eddy/${subjid}__dwi_eddy_corrected.bvec $subj_folder/dwi/bvec_tractoflow
  if [ ! -f $subj_preprocessed_folder/DTI_Metrics/${subjid}__fa.nii.gz ]; then echo "Subject's FA not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/DTI_Metrics/${subjid}__fa.nii.gz $subj_folder/dwi/fa.nii.gz

  # anat:
  if [ ! -f $subj_preprocessed_folder/Register_T1/${subjid}__t1_warped.nii.gz ]; then echo "Subject's T1 not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Register_T1/${subjid}__t1_warped.nii.gz $subj_folder/anat/t1_tractoflow.nii.gz
  if [ ! -f $subj_preprocessed_folder/Segment_Tissues/${subjid}__map_wm.nii.gz ]; then echo "Subject's WM map not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Segment_Tissues/${subjid}__map_wm.nii.gz $subj_folder/anat/wm_map.nii.gz

  # masks:
  if [ ! -f $subj_preprocessed_folder/Resample_B0/${subjid}__b0_mask_resampled.nii.gz ]; then echo "Subject's b0 bet mask not found"; exit 1; fi
  ln -s $subj_preprocessed_folder/Resample_B0/${subjid}__b0_mask_resampled.nii.gz $subj_folder/masks/b0_bet_mask_resampled.nii.gz

  # bundles:
  for dir in $subj_preprocessed_folder/*Tracking
  do
    if [ ! -f $dir/${subjid}__*tracking*.trk ]; then echo "Subject's tractogram not found"; exit 1; fi
    ln -s $dir/${subjid}__*tracking*.trk $subj_folder/bundles/tractoflow_wholebrain.trk
  done
done

echo "We have organized tractoflow results from processed into dwi_ml_ready (dwi, anat, masks)"
