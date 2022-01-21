#!/usr/bin/env python
import os


config_filename = os.path.join(os.path.dirname(__file__),
                               '../tests_on_real_data/config_file_for_tests.json')

hdf5_filename = os.path.join(os.path.dirname(__file__), 'small_hdf5.hdf5')

dwi_ml_folder = os.path.join(os.path.dirname(__file__), 'dwi_ml_ready')
