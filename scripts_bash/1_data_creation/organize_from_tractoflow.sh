# checked!

################################################################################
# This script is used to precess the original HCP data and organize it into the
# expected "dwi_ml_ready" folder architecture, flipping the DWI and bvecs from
# LAS to RAS and creating all necessary masks.
# The dwi_ml_ready/ folder will be created alongside the original/ folder.
#
# Args:
#   working_folder :
#   subjects : A .txt file listing all the subjects ids to process, as expected
#              in the data/ and bundles/ folders.
################################################################################

# Please define your variables here:
# - database_folder = Path to the working folder that contains the original/
#     folder. See dwi_ml.data.creation.description_data_structure.py for more
#     information. Will eventually contain dwi_ml_ready/ and processed/.
# - subjects = the list of ALL subjects. You may choose later which ones will be
#     in your training set/validation set/testing set. Ex: "subj1 subj2"
database_folder=YOUR WORKING FOLDER
subjects_list=SUBJECTS.txt
bundles_list=BUNDLES.txt

# Cleaning path name
database_folder=$(realpath $database_folder)

# Checking if inputs exist
if [ ! -d $database_folder ]; then
  echo "Invalid database_folder argument!"
  exit
fi
if [ ! -d $database_folder/preprocessed ]; then
  echo "There is no preprocessed folder in your database!"
  exit
fi
if [ ! -f $subjects_list ]; then
  echo "Invalid subjects.txt file!"
  exit
fi
if [ ! -f $bundles_list ]; then
  echo "Invalid bundles.txt file!"
  exit
fi

# Preparing dataset subfolders
preprocessed_folder=$database_folder/preprocessed
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
  ln -s $preprocessed_folder/$subjid/Resample_DWI/${subjid}__dwi_resample.nii.gz $subj_folder/dwi/{subject_id}_dwi_preprocessed.nii.gz
  ln -s $preprocessed_folder/$subjid/Eddy/${subjid}__bval_eddy $subj_folder/dwi/{subject_id}_bval_preprocessed
  ln -s $preprocessed_folder/$subjid/Eddy/${subjid}__dwi_eddy_corrected.bvec $subj_folder/dwi/{subject_id}_bvec_preprocessed
  ln -s $preprocessed_folder/$subjid/DTI_Metrics/${subjid}__fa.nii.gz $subj_folder/dwi/{subject_id}_fa.nii.gz

  # anat:
  ln -s $preprocessed_folder/$subjid/Register_T1/${subjid}__t1_warped.nii.gz $subj_folder/anat/{subject_id}_t1.nii.gz
  ln -s $preprocessed_folder/$subjid/Segment_Tissues/${subjid}__map_wm.nii.gz $subj_folder/anat/{subject_id}_wm_map.nii.gz

  # masks:
  ln -s $preprocessed_folder/$subjid/Segment_Tissues/${subjid}__mask_wm.nii.gz $subj_folder/masks/{subject_id}_wm.nii.gz

  # Bundles:
  echo "We organized tractoflow. You could now run and reorganize Recobundles if you wish"

done
