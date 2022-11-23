#!/usr/bin/env python

"""
Organization of the testing data:

├── code_creation
│    ├── code.sh
│    ├── config_file.json
│    ├── empty_subjs_list.txt
│    └── training_subjs.txt
├── dwi_ml_ready
│    └── subjX
│        ├── anat
│        │     └── t1.nii.gz
│        ├── dwi
│        │     └── fa.nii.gz
│        ├── example_bundle
│        │     └── Fornix.trk
│        └── masks
│              └── wm.nii.gz
└── hdf5_file.hdf5

"""
# Values corresponding to the testing data, as of Nov 22, 22.
# Created from Emmanuelle's preprocessed WithReversed data.

TEST_EXPECTED_VOLUME_GROUPS = ['input', 'wm_mask']
TEST_EXPECTED_STREAMLINE_GROUPS = ['streamlines']
TEST_EXPECTED_NB_FEATURES = [2, 1]  # input = t1 + fa. wm_mask = wm.
TEST_EXPECTED_NB_STREAMLINES = [236]  # from the Fornix.trk file
TEST_EXPECTED_NB_SUBJECTS = 1
TEST_EXPECTED_SUBJ_NAMES = ['subjX']
TEST_EXPECTED_MRI_SHAPE = [[138, 166, 134, 2], [138, 166, 134, 1]]
