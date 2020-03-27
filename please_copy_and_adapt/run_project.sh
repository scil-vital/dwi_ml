
########################
# 1. Creating database #
########################
database_dir="my_database"
config_file="my_config_groups.json"
training_subjs="my_training_subjs.txt"
validation_subjs="my_validation_subjs.txt"
name="test1_today"
bundles="AF_left AF_right CC_1 CC_2 FX_left FX_right"
mask="Masks/wm_mask.nii.gz"
space="Space.RASMM"

create_hdf5_dataset.py $database_dir $config_file \
    $training_subjs $validation_subjs \
    --name $name --bundles $bundles --mask $mask --space $space \
    --save_intermediate --logging debug

########################
# 2. Training          #
########################




########################
# 3. Validation        #
########################




########################
# 4. Tracking          #
########################