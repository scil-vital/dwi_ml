
# You can check on Beluga:
# projects/rrg-descotea/datasets/ismrm2015/derivatives,
# and create your own data lists
# subjects: ismrm2015_noArtefact

# Run this from inside dwi_ml.

###########
# Organize data
###########
beluga="YOUR INFOS"
ismrm2015_folder="$beluga/ismrm2015/derivatives"

database_folder="$ismrm2015_folder/noArtefact"
subjects_list="$database_folder/subjects.txt"

rm -r $database_folder/dwi_ml_ready
organize_from_tractoflow_folder=please_copy_and_adapt/
organize_from_tractoflow_folder=../Learn2Track/scripts/
$organize_from_tractoflow_folder/organize_from_tractoflow.sh $database_folder $subjects_list

###########
# Create hdf5
###########
name=ismrm2015_noArtefact_test
logging="info"
mask_for_standardization="masks/b0_bet_mask_resampled.nii.gz"
space="rasmm"
step_size=0.5 #Use step_size for now. Dipy's compress_streamlines seems to have memory issues.
training_subjs="$database_folder/subjects_for_ML_training.txt"
validation_subjs="$database_folder/subjects_for_ML_validation.txt"
config_file="please_copy_and_adapt/config_file_example.json"

echo ismrm2015_noArtefact > $training_subjs
echo fake_subj2 > $validation_subjs
cp -r $database_folder/dwi_ml_ready/ismrm2015_noArtefact/ $database_folder/dwi_ml_ready/fake_subj2

create_hdf5_dataset.py --force --name $name --save_intermediate \
        --std_mask $mask_for_standardization --independent_modalities True \
        --logging $logging --space $space \
        --enforce_files_presence True --step_size $step_size \
        $database_folder $config_file $training_subjs $validation_subjs

###########
# Tests on dataset and batch sampler
###########
hdf5_filename="$database_folder/hdf5/$name.hdf5"
ref="$database_folder/dwi_ml_ready/ismrm2015_noArtefact/anat/t1_tractoflow.nii.gz"
test_tractograms_path="$database_folder/hdf5/tests/"

mkdir $test_tractograms_path

tests/test_multisubjectdataset_creation_from_hdf5.py $hdf5_filename
tests/test_batch_sampler_iter.py $hdf5_filename
tests/test_batch_sampler_load_batch.py $hdf5_filename $ref $test_tractograms_path

# check results and then:
rm $test_tractograms_path/test_batch1*

###########
# Running training on chosen database:
# This will stop with error "optimizer got an empty parameter list" at line
# self.optimizer = torch.optim.Adam(self.model.parameters()) in the trainer.
# Further testing was done with Learn2track's model.
###########
mkdir $database_folder/experiments
python please_copy_and_adapt/train_model.py --logging debug \
      --input_group 'input' --target_group 'streamlines' \
      --hdf5_filename $database_folder/hdf5/ismrm2015_noArtefact_test.hdf5 \
      --parameters_filename please_copy_and_adapt/training_parameters.yaml \
      --experiment_name test_experiment1 $database_folder/experiments
