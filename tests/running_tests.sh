
# You can check on Beluga:
# projects/rrg-descotea/datasets/ismrm2015/derivatives,
# and create your own data lists
# subjects: ismrm2015_noArtefact

###########
# Organize data
###########
beluga="YOUR INFOS"
ismrm2015_folder="$beluga/ismrm2015/derivatives"

database_folder="$ismrm2015_folder/noArtefact"
subjects_list="$ismrm2015_folder/noArtefact/subjects.txt"
training_subjs="$ismrm2015_folder/noArtefact/ML_studies/subjects_for_ML_training.txt"
validation_subjs="$ismrm2015_folder/noArtefact/ML_studies/subjects_for_ML_validation.txt"
config_file="$ismrm2015_folder/noArtefact/ML_studies/config_file.json"
# My config file is:
#{
#    "input": {
#	        "type": "volume",
#       	"files": ["dwi/dwi_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz", "anat/t1_tractoflow.nii.gz"]
#	     }
#}
organize_from_tractoflow.sh $database_folder $subjects_list

###########
# Create hdf5
###########
name=ismrm2015_noArtefact_test
option_bundles="" #empty = takes everything
logging="info"
mask_for_standardization="masks/wm_mask.nii.gz"
space="rasmm"
step_size=0.5 #Use step_size for now. Dipy's compress_streamlines seems to have memory issues.
create_hdf5_dataset.py --force --name $name --std_mask $mask_for_standardization \
        $option_bundles --logging $logging --space $space $database_folder \
        --enforce_bundles_presence True --step_size $step_size \
        $config_file $training_subjs $validation_subjs

###########
# Tests
###########
hdf5_filename="$ismrm2015_folder/noArtefact/hdf5/$name.hdf5"
tests/test_multisubjectdataset_creation_from_hdf5.py $hdf5_filename
tests/test_batch_sampler_iter.py $hdf5_filename
tests/test_batch_sampler_load_batch.py $hdf5_filename
