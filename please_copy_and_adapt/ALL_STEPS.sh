database_folder="my_database"
all_subjs="my_subjs.txt"
recobundles_folder=RecobundlesX/multi_bundles

# If you need to prepare your data:
bash organize_from_tractoflow.sh $database_folder $all_subjs
bash organize_from_recobundles.sh $database_folder $recobundles_folder $all_subjs

########################
# 1. Creating database #
########################
config_file="my_config_groups.json"
training_subjs="my_training_subjs.txt"
validation_subjs="my_validation_subjs.txt"
name="test1"
mask="masks/wm_mask.nii.gz"
space="rasmm"

create_hdf5_dataset.py --name $name --std_mask $mask \
    --space $space --save_intermediate --logging debug \
    --enforce_files_presence True \
    $database_folder/dwi_ml_ready $database_folder $config_file \
    $training_subjs $validation_subjs

########################
# 2. Training          #
########################
yaml_filename=training_parameters.yaml
hdf5_filename="$database_folder/hdf5/$name.hdf5"

train_model.py --yaml_parameters $yaml_filename --hdf5_file $hdf5_file \
    $my_path my_experiment

# Run from checkpoint
    train_model.py --resume --override_checkpoint_max_epochs 20 \
    $my_path my_experiment

visualize_logs.py $my_path/my_experiment


########################
# 3. Tracking          #
########################
