
tests/test_direction_getter_models.py

# Create any hdf5 dataset. Ex, use please_copy_and_adapt/ALL_STEPS.sh

tests/test_multisubjectdataset_creation_from_hdf5.py $hdf5_filename
tests/test_batch_sampler_iter.py $hdf5_filename

# Note. This also tests neighborhoods in the model:
tests/test_batch_sampler_load_batch.py $hdf5_filename $ref $test_tractograms_path

# check results and then:
rm $test_tractograms_path/test_batch*
