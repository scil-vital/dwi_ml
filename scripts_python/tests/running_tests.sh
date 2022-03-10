
tests/test_direction_getter_models.py
tests/test_previous_dirs.py


# More complex tests need data:
# Create any hdf5 dataset. Ex, use please_copy_and_adapt/ALL_STEPS.sh
hdf5_filename=my_file
tests/test_multisubjectdataset_creation_from_hdf5.py "$hdf5_filename"
tests/test_batch_sampler_iter.py "$hdf5_filename" 'input'

# Note. This also tests neighborhoods in the model:
ref=my_ref
test_tractograms_path='./'
tests/test_batch_sampler_load_batch.py "$hdf5_filename" 'input' "$ref" "$test_tractograms_path"
# check results and then:
rm "$test_tractograms_path"/test_batch*
