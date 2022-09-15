


# Choose study
#########
database_folder=MY_PATH


# Organize your files.
#########


# Create hdf5 dataset
#########
space='rasmm'
name='my_hdf5_database'
mask="mask/*__mask_wm.nii.gz"

dwi_ml_ready_folder=my_path/dwi_ml_ready
hdf5_folder=my_path
config_file=my_config_file
training_subjs=file1.txt
validation_subjs=file2.txt
testing_subjs=file3.txt

l2t_create_hdf5_dataset.py --force --name $name --std_mask $mask \
        --logging info --space $space --enforce_files_presence True \
        $dwi_ml_ready_folder $hdf5_folder $config_file \
        $training_subjs $validation_subjs $testing_subjs

# Train model
############
batch_size=100
batch_size_units='nb_streamlines'
max_batches_per_epoch=10
max_epochs=2
experiment_name=test_experiment
input_group_name='input'
streamline_group_name='streamlines'

experiment_folder=my_path
hdf5_file=my_path/my_hdf5_database.hdf5

l2t_train_model.py --logging 'info' --lazy \
        --nb_previous_dirs 3 \
        --direction_getter_key cosine-regression --normalize_direction \
        --batch_size $batch_size --batch_size_units $batch_size_units \
        --max_epochs $max_epochs --max_batches_per_epoch $max_batches_per_epoch \
        $experiment_folder $experiment_name $hdf5_file \
        $input_group_name $streamline_group_name

# Run from checkpoint
l2t_resume_training_from_checkpoint.py --logging info  \
    --new_max_epochs 3 $experiment_folder $experiment_name

visualize_logs.py $experiment_folder/$experiment_name


# Track from model
##############
tracking_mask=
seeding_mask=
subj_id=
out_tractogram=my_tractogram
algo='det'

l2t_track_from_model.py \
        $tracking_mask $seeding_mask --input_from_hdf5 'input' \
        --subj_id $subj_id --hdf5_file $hdf5_file \
         --logging info \
        $experiment_folder/$experiment_name $out_tractogram $algo
