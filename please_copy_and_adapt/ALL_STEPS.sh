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
bundles="AF_left AF_right CC_1 CC_2 FX_left FX_right"
# OR bundles=$(<bundles.txt) to read a list of bundles from a text file.
mask="masks/wm_mask.nii.gz"
space="rasmm"
enforce_bundles_presence=True # True if you want the script to stop if a bundle is missing for some subject

create_hdf5_dataset.py --name $name --bundles $bundles --std_mask $mask \
    --space $space --save_intermediate --logging debug \
    --enforce_bundles_presence $enforce_bundles_presence
    $database_folder $config_file $training_subjs $validation_subjs

########################
# 2. Training          #
########################

yaml_file=training_parameters.yaml

train_model.py $yaml_file

########################
# 3. Validation        #
########################




########################
# 4. Tracking          #
########################
