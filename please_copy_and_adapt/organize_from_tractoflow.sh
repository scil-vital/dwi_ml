################################################################################
# This script will create symlinks in dwi_ml_ready, pointing to your data      #
# from tractoflow for each subject:                                            #
#    - Resample/dwi_resample (which is the last dwi output from tractoflow.)   #
#         Will be copied in dwi/dwi_tractoflow.                                #
#    - Eddy/bval_eddy. Will be copied in dwi/bval_tractoflow. Same for bvec.   #
#    - DTI_metrics/fa. Will be copied to dwi/fa.                               #
#    - Register_T1/t1_warp (which is the last T1). Will be copied to anat/t1.  #
#    - Segment_Tissues/map_wm. Will be copied to anat/wm_map                   #
#    - Segment_Tissues/mask_wm. Will be copied to masks/wm                     #
#                                                                              #
# If you need something else for your model, you can modify this script.       #
#                                                                              #
# See our doc for more information                                             #
# (https://dwi-ml.readthedocs.io/en/latest/data_organization.html#ref-organization).
# We suppose that you have a "preprocessed" folder that contains RecobundlesX  #
# results folder for each subject.                                             #
################################################################################

# =====================================#
#  VARIABLES TO BE DEFINED BY THE USER #
# =====================================#
# - database_folder = Path to the working folder that contains the original/
#     folder. Will eventually contain dwi_ml_ready/.
# - subjects = The list of ALL subjects. You may choose later which ones will be
#     in your training set/validation set/testing set. One subject per line.
database_folder=YOUR WORKING FOLDER
subject_list=SUBJECTS.txt

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
while IFS= read -r subjid; do
  echo "Reorganizing subject $subjid"
  subj_preprocessed_folder=$database_folder/preprocessed/$subjid
  subj_folder=$dwi_ml_ready_folder/$subjid
  mkdir $subj_folder
  mkdir $subj_folder/anat
  mkdir $subj_folder/dwi
  mkdir $subj_folder/masks

  echo "creating symlinks"
  # dwi:
  ln -s $subj_preprocessed_folder/Resample_DWI/${subjid}__dwi_resample.nii.gz $subj_folder/dwi/${subjid}_dwi_tractoflow.nii.gz
  ln -s $subj_preprocessed_folder/Eddy/${subjid}__bval_eddy $subj_folder/dwi/${subjid}_bval_tractoflow
  ln -s $subj_preprocessed_folder/Eddy/${subjid}__dwi_eddy_corrected.bvec $subj_folder/dwi/${subjid}_bvec_tractoflow
  ln -s $subj_preprocessed_folder/DTI_Metrics/${subjid}__fa.nii.gz $subj_folder/dwi/${subjid}_fa.nii.gz

  # anat:
  ln -s $subj_preprocessed_folder/Register_T1/${subjid}__t1_warped.nii.gz $subj_folder/anat/${subjid}_t1.nii.gz
  ln -s $subj_preprocessed_folder/Segment_Tissues/${subjid}__map_wm.nii.gz $subj_folder/anat/${subjid}_wm_map.nii.gz

  # masks:
  ln -s $subj_preprocessed_folder/Segment_Tissues/${subjid}__mask_wm.nii.gz $subj_folder/masks/${subjid}_wm.nii.gz

  # Bundles:

done < $subject_list

echo "We have organized tractoflow results into dwi_ml (dwi, anat, masks)".
echo "We do not raise warnings if one file is not found. Please check that all data was indeed found."
echo "Ex: 'for subj in dwi_ml_ready/*; do ls \$subj/dwi/*bvec*; done'"
echo "Hint: We have not treated any bundles."
echo "      As a next step, you could now run Recobundles and use organize_from_recobundles (to come)"