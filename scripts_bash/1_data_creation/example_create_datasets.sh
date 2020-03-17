###
# Script to create the various processed datasets
# Args:
#   dataset_folder : Path to the folder containing the raw/ dataset folder
###
dataset_folder="$1"
config_folder=$(dirname "$0")

# fODF 3-peaks datasets
create_hdf5_dataset.py ${dataset_folder} "$config_folder"/trainset_config.json --name HCP105_trainset_peaks3 --logging info fodf-peaks --sh-order 6 --n-peaks 3
create_hdf5_dataset.py ${dataset_folder} "$config_folder"/validset_config.json --name HCP105_validset_peaks3 --logging info fodf-peaks --sh-order 6 --n-peaks 3
create_hdf5_dataset.py ${dataset_folder} "$config_folder"/testset_config.json --name HCP105_testset_peaks3 --logging info fodf-peaks --sh-order 6 --n-peaks 3

# fODF SH datasets
#create_hdf5_dataset.py ${dataset_folder} trainset_config.json --name HCP105_trainset_sh6 fodf-sh --sh-order 6
#create_hdf5_dataset.py ${dataset_folder} validset_config.json --name HCP105_validset_sh6 fodf-sh --sh-order 6
#create_hdf5_dataset.py ${dataset_folder} testset_config.json --name HCP105_testset_sh6 fodf-sh --sh-order 6
