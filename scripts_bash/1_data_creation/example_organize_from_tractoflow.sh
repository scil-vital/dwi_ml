# checked!

################################################################################
# This script is used to reorganize tractoflow results into the dwi_ml_ready
# folder. You should copy this script into your project and modify according to
# your needs.
# See dwi_ml.data.hdf5_creation.description_data_structure.py for more information.
# We suppose that you have a "preprocess" folder that contains tractoflow's
# results.
# This script will create the dwi_ml_folder and subfolders and copy the main
# data from tractoflow:
#    - dwi_resample (the last dwi output from tractoflow. tractoflow.)
#    - bval_eddy, bvec_eddy:  tractoflow bval and bvec.
#    - The FA
#    - t1_warp (the last T1: resampled, registered, tractoflow).
#    - map_wm and mask_wm.
# If you need something else for your model, you can modify this script.
################################################################################

# =====================================#
#  VARIABLES TO BE DEFINED BY THE USER #
# =====================================#
# - database_folder = Path to the working folder that contains the original/
#     folder. Will eventually contain dwi_ml_ready/.
# - subjects = The list of ALL subjects. You may choose later which ones will be
#     in your training set/validation set/testing set. Ex: "subj1 subj2"
# - tractoflow_folder = Name of your tractoflow folder inside preprocessed.
database_folder=YOUR WORKING FOLDER
subjects_list=SUBJECTS.txt
tractoflow_folder=TRACTOFLOW

# =====================================#
#            MAIN SCRIPT               #
# =====================================#
# Cleaning path name
database_folder=$(realpath $database_folder)
tractoflow_folder=$database_folder/$tractoflow_folder

# Checking if inputs exist
if [ ! -d $database_folder ]; then
  echo "Invalid database_folder argument!"
  exit
fi
if [ ! -d $tractoflow_folder ]; then
  echo "There is no tractoflow folder in your preprocess folder!"
  exit
fi
if [ ! -f $subjects_list ]; then
  echo "Invalid subjects txt file!"
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
for subjid in $(<"$subjects"); do
  echo "Reorganizing subject $subjid"
  subj_folder=$dwi_ml_ready_folder/$subjid
  mkdir $subj_folder
  mkdir $subj_folder/dwi
  mkdir $subj_folder/bundles
  mkdir $subj_folder/masks
  mkdir $subj_folder/masks/bundles
  mkdir $subj_folder/masks/endpoints

  echo "creating symlinks"
  # dwi:
  ln -s $tractoflow_folder/$subjid/Resample_DWI/${subjid}__dwi_resample.nii.gz $subj_folder/dwi/{subject_id}_dwi_tractoflow.nii.gz
  ln -s $tractoflow_folder/$subjid/Eddy/${subjid}__bval_eddy $subj_folder/dwi/{subject_id}_bval_tractoflow
  ln -s $tractoflow_folder/$subjid/Eddy/${subjid}__dwi_eddy_corrected.bvec $subj_folder/dwi/{subject_id}_bvec_tractoflow
  ln -s $tractoflow_folder/$subjid/DTI_Metrics/${subjid}__fa.nii.gz $subj_folder/dwi/{subject_id}_fa.nii.gz

  # anat:
  ln -s $tractoflow_folder/$subjid/Register_T1/${subjid}__t1_warped.nii.gz $subj_folder/anat/{subject_id}_t1.nii.gz
  ln -s $tractoflow_folder/$subjid/Segment_Tissues/${subjid}__map_wm.nii.gz $subj_folder/anat/{subject_id}_wm_map.nii.gz

  # masks:
  ln -s $tractoflow_folder/$subjid/Segment_Tissues/${subjid}__mask_wm.nii.gz $subj_folder/masks/{subject_id}_wm.nii.gz

  # Bundles:
  echo "We have organized tractoflow results into dwi_ml (dwi, anat, masks)".
  echo "We have not treated any bundles."
  echo "Hint: As a next step, you could now run Recobundles and use example_organize_from_recobundles (to come)"

done
