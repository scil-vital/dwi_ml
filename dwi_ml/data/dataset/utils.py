# -*- coding: utf-8 -*-
import logging

from dwi_ml.data.dataset.multi_subject_containers import MultiSubjectDataset
from dwi_ml.experiment_utils.timer import Timer


def prepare_multisubjectdataset(args, load_training=True, load_validation=True,
                                load_testing=True,
                                log_level=logging.root.level):
    """
    Instantiates a MultiSubjectDataset AND loads data.

    Params
    ------
    args: Namespace
        Must contain 'hdf5_File, 'lazy' and 'cache_size'
    """
    with Timer("\nPreparing datasets", newline=True, color='blue'):
        dataset = MultiSubjectDataset(
            args.hdf5_file, lazy=args.lazy, cache_size=args.cache_size,
            log_level=log_level)
        dataset.load_data(load_training, load_validation, load_testing)

        logging.info("Number of subjects loaded: \n"
                     "      Training: {}\n"
                     "      Validation: {}\n"
                     "      Testing: {}"
                     .format(dataset.training_set.nb_subjects,
                             dataset.validation_set.nb_subjects,
                             dataset.testing_set.nb_subjects))

    return dataset
