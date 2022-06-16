database_folder=PATH_TO_MY_PROCESSED_FILES/
dwi_ml_folder=PATH_TO_ORGANIZED_FILES/dwi_ml_ready/


# If you need to prepare your data:
all_subjs="my_subjs.txt"
recobundles_folder=RecobundlesX/multi_bundles
bash organize_from_tractoflow.sh $database_folder $all_subjs
bash organize_from_recobundles.sh $database_folder $recobundles_folder $all_subjs

########################
# 1. Creating database #
########################
config_file="my_config_groups.json"
training_subjs="my_training_subjs.txt"
validation_subjs="my_validation_subjs.txt"
testing_subjs="my_testing_subjs.txt"
name="test1"
std_mask="masks/wm.nii.gz"
space="rasmm"  # {rasmm,vox,voxmm}

hdf5_file=YOUR_PATH/hdf5_test1.hdf5
dwiml_create_hdf5_dataset.py --logging debug \
        --std_mask $std_mask --space $space --compress \
        "$dwi_ml_folder" "$hdf5_file" \
        "$config_file" "$training_subjs" "$validation_subjs" "$testing_subjs"

########################
# 2. Training          #
########################
model_options="--sphere_radius 1"
streamline_options=" --split_ratio 0.5 --reverse_ratio 0.5 --noise_gaussian_size 0.1 --step_size 0.5"
batch_options="--batch_size 200 --batch_size_units nb_streamlines"
epoch_options="--max_epochs 2 --max_batches_per_epoch 10"

experiments_path=MY_EXPERIMENTS/
experiment_name=test
input_name='input'
streamlines_name='streamlines'
dwi_ml_train_model.py --logging INFO --use_gpu \
    --comet_workspace emmarenauld --comet_project learn2track \
    $model_options $streamline_options \
    $batch_options $epoch_options \
    $experiments_path $experiment_name $hdf5_file \
    $input_name $streamlines_name


#========== Run from checkpoint
dwi_ml_resume_training_from_checkpoint.py --logging INFO  \
     --new_max_epochs 10 "$experiments_path" $experiment_name

# ========== Check results
dwiml_visualize_logs.py $experiments_path/$experiment_name


########################
# 3. Tracking          #
########################
subj_id=a_test_subject

mkdir TRACKING_PATH/$subj_id
out_tractogram=TRACKING_PATH/$subj_id/tractogram.trk

tracking_mask_group=wm_mask
seeding_mask_group=wm_mask
input_group=input

dwi_ml_track_from_model.py -f \
    --algo 'det' --step 0.5 --min_length 10 --max_length 300 \
    --rk_order 1 --theta 45 --processes 1 --rk_order 1 \
    --nt 99 --logging DEBUG --rng_seed 1234 \
    --cache_size 1 --lazy \
    --use_gpu --simultaneous_tracking 100 \
    $experiments_path/$experiment_name/ $hdf5_file $subj_id $out_tractogram \
    $seeding_mask_group $tracking_mask_group $input_group