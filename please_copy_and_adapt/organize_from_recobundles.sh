################################################################################
# This script will create symlinks in dwi_ml_ready, pointing to your bundles   #
# from RecobundlesX for each subject.                                          #
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
# -dwi_ml_ready_folder = Path to the database, which should contain a
#     preprocessed folder and a dwi_ml_ready folder.
# -recobundles_name = Name of the recobundlesX folder for each
#     subject. Ex: RecobundlesX/multi_bundles
# -subjects = The list of ALL subjects. You may choose later which ones will be
#     in your training set/validation set/testing set. One subject per line.
database_folder=YOUR DATABASE FOLDER
recobundles_name=RecobundlesX/multi_bundles
subject_list=SUBJECTS.txt

# =====================================#
#            MAIN SCRIPT               #
# =====================================#
# Cleaning path name
database_folder=$(realpath $database_folder)
preprocessed_folder=$database_folder/preprocessed
dwi_ml_ready_folder=$database_folder/dwi_ml_ready

# Checking if files exist
if [ ! -d $preprocessed_folder ]; then
  echo "Preprocessed folder not found! ($preprocessed_folder)!"
  exit
fi
if [ ! -d $dwi_ml_ready_folder ]; then
  echo "dwi_ml_ready folder not found! ($dwi_ml_ready_folder)!"
  exit
fi
if [ ! -f $subject_list ]; then
  echo "Invalid subjects txt file! ($subject_list)"
  exit
fi

# Reorganizing all subjects
while IFS= read -r subjid; do
  echo "Reorganizing subject $subjid"
  subj_folder=$dwi_ml_ready_folder/$subjid
  recobundles_folder=$preprocessed_folder/$subjid/$recobundles_name
  mkdir $subj_folder/bundles

  echo "creating symlinks"
  # bundles:
  cd $recobundles_folder
  for bundle in *.trk
  do
    ln -s $recobundles_folder/$bundle $subj_folder/bundles/${subjid}_recobundlesX_$bundle
  done
done