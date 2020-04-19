
########################
# 1. Creating database #
########################
dwi_ml_ready_folder="my_database"
config_file="my_config_groups.json"
training_subjs="my_training_subjs.txt"
validation_subjs="my_validation_subjs.txt"
name="test1"
bundles="AF_left AF_right CC_1 CC_2 FX_left FX_right"
# OR bundles=$(<bundles.txt) to read a list of bundles from a text file.
mask="masks/wm_mask.nii.gz"
space="Space.RASMM"  # {Space.RASMM,Space.VOX,Space.VOXMM}

create_hdf5_dataset.py --name $name --bundles $bundles --mask $mask \
    --space $space --save_intermediate --logging debug \
    $dwi_ml_ready_folder $config_file $training_subjs $validation_subjs

########################
# 2. Training          #
########################

 create_hdf5_dataset.py [-h] [--step_size STEP_SIZE] [--name NAME]
                              [--bundles BUNDLES [BUNDLES ...]] [--mask MASK]
                              [--space {Space.RASMM,Space.VOX,Space.VOXMM}]
                              [--logging {error,warning,info,debug}]
                              [--save_intermediate] [-f]
                              path config_file training_subjs validation_subjs


########################
# 3. Validation        #
########################




########################
# 4. Tracking          #
########################
